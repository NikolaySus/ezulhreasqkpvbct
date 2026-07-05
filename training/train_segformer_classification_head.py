from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.train_classification_head import (
    CLASS_NAMES,
    ImageRecord,
    binary_metrics,
    class_pos_weight,
    limit_records,
    load_manifest,
    plot_history,
    seed_worker,
    set_seed,
    sigmoid,
    sliding_positions,
    write_history,
    write_test_report,
)


Image.MAX_IMAGE_PIXELS = None

NUM_SEGMENTATION_CLASSES = 4
PATCH_SIZE = 512
TRAIN_PATCHES_PER_IMAGE = 16
EVAL_PATCH_STRIDE = 192
GRAY_MEAN = 0.449
GRAY_STD = 0.226
ROT_RANGE_DEG = (-180.0, 180.0)
SCALE_RANGE = (0.85, 1.15)
TRAIN_MIN_EXTRACT = int(math.ceil(PATCH_SIZE * math.sqrt(2.0) / SCALE_RANGE[0]) + 4)
SEGMENTATION_COLORS = np.array(
    [
        [255, 0, 0],
        [255, 255, 0],
        [0, 80, 255],
        [128, 0, 64],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class PatchRecord:
    image_index: int
    x0: int
    y0: int


class SegformerClassificationHead(nn.Module):
    def __init__(self, in_channels: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.linear(self.dropout(pooled)).squeeze(1)


class SegformerClassificationModel(nn.Module):
    def __init__(self, segmentation_model: nn.Module, classification_head: nn.Module) -> None:
        super().__init__()
        self.segmentation_model = segmentation_model
        self.classification_head = classification_head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.segmentation_model.encoder(images)
        return self.classification_head(features[-1])

    def forward_segmentation_and_classification(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.segmentation_model.encoder(images)
        decoder_output = self.segmentation_model.decoder(features)
        segmentation_logits = self.segmentation_model.segmentation_head(decoder_output)
        classification_logits = self.classification_head(features[-1])
        return segmentation_logits, classification_logits


class RandomPatchDataset(Dataset):
    def __init__(self, records: list[ImageRecord], patches_per_image: int, seed: int) -> None:
        self.records = list(records)
        random.Random(seed).shuffle(self.records)
        self.patches_per_image = patches_per_image
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.transforms = train_transforms()

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index = index // self.patches_per_image
        record = self.records[image_index]
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        image = cached_load_gray(self.image_cache, record.image_path, max_cached_images=4)
        patch = sample_safe_rotated_patch(image, rng)
        patch = self.transforms(image=patch)["image"]
        return preprocess_gray(patch), torch.tensor(float(record.label_id), dtype=torch.float32)


class EvalPatchDataset(Dataset):
    def __init__(self, records: list[ImageRecord], max_patches_per_image: int | None = None) -> None:
        self.records = records
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.patch_records: list[PatchRecord] = []
        for image_index, record in enumerate(records):
            with Image.open(record.image_path) as image:
                width, height = image.size
            patches = [
                PatchRecord(image_index=image_index, x0=x0, y0=y0)
                for y0 in sliding_positions(height, PATCH_SIZE, EVAL_PATCH_STRIDE)
                for x0 in sliding_positions(width, PATCH_SIZE, EVAL_PATCH_STRIDE)
            ]
            if max_patches_per_image is not None and len(patches) > max_patches_per_image:
                picks = np.linspace(0, len(patches) - 1, max_patches_per_image).round().astype(int)
                patches = [patches[index] for index in picks]
            self.patch_records.extend(patches)

    def __len__(self) -> int:
        return len(self.patch_records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_record = self.patch_records[index]
        image_record = self.records[patch_record.image_index]
        image = cached_load_gray(self.image_cache, image_record.image_path, max_cached_images=4)
        patch = crop_patch(image, patch_record.x0, patch_record.y0, PATCH_SIZE)
        return (
            preprocess_gray(patch),
            torch.tensor(float(image_record.label_id), dtype=torch.float32),
            torch.tensor(patch_record.image_index, dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary ore-classification head on SegFormer features.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("training-artifacts/classification_dataset/v1"))
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=None,
        help="SegFormer segmentation checkpoint with model_state_dict from ex6.ipynb.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training-artifacts/segformer_classification_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--head-only-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patches-per-image", type=int, default=TRAIN_PATCHES_PER_IMAGE)
    parser.add_argument("--eval-max-patches-per-image", type=int, default=64)
    parser.add_argument("--drift-max-patches", type=int, default=128)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drift-threshold", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Smoke-test limiter.")
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip training and evaluate an existing SegFormer classification checkpoint on the test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.segmentation_checkpoint is None:
        args.segmentation_checkpoint = default_segformer_checkpoint_path()
    set_seed(args.seed)

    run_id = args.run_name or time.strftime("segformer_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    checkpoint_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    demo_dir = run_dir / "demo"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(args.dataset_dir / "manifest.csv")
    if args.max_images_per_class is not None:
        records = limit_records(records, args.max_images_per_class)
    split_records = {split: [record for record in records if record.split == split] for split in ("train", "val", "test")}
    if any(not split_records[split] for split in split_records):
        raise RuntimeError(f"Empty split after loading {args.dataset_dir / 'manifest.csv'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_model = build_segformer_model(args.segmentation_checkpoint, device)
    classification_head = SegformerClassificationHead().to(device)
    model = SegformerClassificationModel(segmentation_model, classification_head).to(device)

    freeze_segmentation_decoder(segmentation_model)
    configure_trainable_parameters(segmentation_model, classification_head, train_final_stage=False)
    optimizer = make_optimizer(classification_head, segmentation_model, args, train_final_stage=False)

    train_loader = DataLoader(
        RandomPatchDataset(split_records["train"], args.patches_per_image, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        EvalPatchDataset(split_records["val"], args.eval_max_patches_per_image),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        EvalPatchDataset(split_records["test"], args.eval_max_patches_per_image),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    drift_loader = DataLoader(
        make_drift_dataset(split_records["val"], args.drift_max_patches),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(split_records["train"]).to(device))

    write_run_config(run_dir, args, split_records, device)

    if args.evaluate_checkpoint is not None:
        load_training_checkpoint(model, args.evaluate_checkpoint, device)
        test_loss, test_metrics, test_payload = evaluate(
            model,
            test_loader,
            criterion,
            split_records["test"],
            device,
            return_payload=True,
        )
        report = write_test_report(
            run_dir,
            plots_dir,
            test_loss,
            test_metrics,
            test_payload,
            split_records["test"],
            checkpoint_path=args.evaluate_checkpoint,
        )
        generate_demo_outputs(model, split_records["test"], test_payload, demo_dir, device)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    baseline_segmentation = collect_segmentation_predictions(segmentation_model, drift_loader, device)
    history: list[dict[str, float | int | str]] = []
    best_score: tuple[float, float] | None = None
    best_path: Path | None = None
    recommended_score: tuple[float, float] | None = None

    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_only_epochs + 1:
            configure_trainable_parameters(segmentation_model, classification_head, train_final_stage=True)
            optimizer = make_optimizer(classification_head, segmentation_model, args, train_final_stage=True)

        train_final_stage = epoch > args.head_only_epochs
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, train_final_stage=train_final_stage)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, split_records["val"], device)
        drift = segmentation_drift(segmentation_model, drift_loader, baseline_segmentation, device)

        row = {
            "epoch": epoch,
            "phase": "head_only" if epoch <= args.head_only_epochs else "segformer_stage4_finetune",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "segmentation_drift": drift,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        save_checkpoint(checkpoint_path, model, optimizer, epoch, history, args, split_records)

        score = (float(val_metrics["roc_auc"]), float(val_metrics["f1"]))
        if best_score is None or score > best_score:
            best_score = score
            best_path = checkpoint_path
            save_checkpoint(checkpoint_dir / "best_val_auc.pth", model, optimizer, epoch, history, args, split_records)
        if drift <= args.drift_threshold and (recommended_score is None or score > recommended_score):
            recommended_score = score
            save_checkpoint(checkpoint_dir / "recommended.pth", model, optimizer, epoch, history, args, split_records)

        write_history(run_dir / "history.csv", history)
        plot_history(history, plots_dir)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"drift={drift:.4f}"
        )

    recommended_checkpoint = checkpoint_dir / "recommended.pth"
    if not recommended_checkpoint.exists() and best_path is not None:
        load_training_checkpoint(model, best_path, device)
        save_checkpoint(
            recommended_checkpoint,
            model,
            optimizer,
            int(history[-1]["epoch"]),
            history,
            args,
            split_records,
        )

    load_training_checkpoint(model, recommended_checkpoint, device)
    test_loss, test_metrics, test_payload = evaluate(
        model,
        test_loader,
        criterion,
        split_records["test"],
        device,
        return_payload=True,
    )
    report = write_test_report(
        run_dir,
        plots_dir,
        test_loss,
        test_metrics,
        test_payload,
        split_records["test"],
        checkpoint_path=recommended_checkpoint,
        best_path=best_path,
    )
    generate_demo_outputs(model, split_records["test"], test_payload, demo_dir, device)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Run artifacts written to {run_dir}")


def default_segformer_checkpoint_path() -> Path:
    container_path = Path("/model-artifacts/ml-days-2/segformer/epoch_028.pth")
    return container_path if container_path.exists() else Path("model-artifacts/ml-days-2/segformer/epoch_028.pth")


def build_segformer_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=1,
        classes=NUM_SEGMENTATION_CLASSES,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def freeze_segmentation_decoder(segmentation_model: nn.Module) -> None:
    for module in (segmentation_model.decoder, segmentation_model.segmentation_head):
        for parameter in module.parameters():
            parameter.requires_grad = False


def configure_trainable_parameters(
    segmentation_model: nn.Module,
    classification_head: nn.Module,
    *,
    train_final_stage: bool,
) -> None:
    for parameter in segmentation_model.parameters():
        parameter.requires_grad = False
    if train_final_stage:
        for name in ("patch_embed4", "block4", "norm4"):
            module = getattr(segmentation_model.encoder, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True
    for parameter in classification_head.parameters():
        parameter.requires_grad = True


def make_optimizer(
    classification_head: nn.Module,
    segmentation_model: nn.Module,
    args: argparse.Namespace,
    *,
    train_final_stage: bool,
) -> torch.optim.Optimizer:
    groups: list[dict[str, object]] = [{"params": classification_head.parameters(), "lr": args.head_lr}]
    if train_final_stage:
        final_stage_parameters = []
        for name in ("patch_embed4", "block4", "norm4"):
            module = getattr(segmentation_model.encoder, name, None)
            if module is not None:
                final_stage_parameters.extend(list(module.parameters()))
        if final_stage_parameters:
            groups.append({"params": final_stage_parameters, "lr": args.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_epoch(
    model: SegformerClassificationModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train_final_stage: bool,
) -> float:
    model.eval()
    model.classification_head.train()
    if train_final_stage:
        for name in ("patch_embed4", "block4", "norm4"):
            module = getattr(model.segmentation_model.encoder, name, None)
            if module is not None:
                module.train()

    running_loss = 0.0
    samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * images.size(0)
        samples += images.size(0)
    return running_loss / max(samples, 1)


@torch.inference_mode()
def evaluate(
    model: SegformerClassificationModel,
    loader: DataLoader,
    criterion: nn.Module,
    records: list[ImageRecord],
    device: torch.device,
    *,
    return_payload: bool = False,
) -> tuple[float, dict[str, float]] | tuple[float, dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    patch_logits: dict[int, list[float]] = {index: [] for index in range(len(records))}
    running_loss = 0.0
    samples = 0
    for images, labels, image_indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += float(loss.item()) * images.size(0)
        samples += images.size(0)
        for image_index, logit in zip(image_indices.tolist(), logits.detach().cpu().tolist()):
            patch_logits[image_index].append(float(logit))

    y_true = np.array([record.label_id for record in records], dtype=np.int64)
    logits = np.array([np.mean(patch_logits[index]) for index in range(len(records))], dtype=np.float32)
    probabilities = sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = binary_metrics(y_true, predictions, probabilities)
    mean_loss = running_loss / max(samples, 1)
    if return_payload:
        return mean_loss, metrics, {"y_true": y_true, "probabilities": probabilities, "predictions": predictions}
    return mean_loss, metrics


@torch.inference_mode()
def collect_segmentation_predictions(
    segmentation_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[np.ndarray]:
    segmentation_model.eval()
    predictions: list[np.ndarray] = []
    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        logits = segmentation_model(images)
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return predictions


@torch.inference_mode()
def segmentation_drift(
    segmentation_model: nn.Module,
    loader: DataLoader,
    baseline_predictions: list[np.ndarray],
    device: torch.device,
) -> float:
    current_predictions = collect_segmentation_predictions(segmentation_model, loader, device)
    changed = 0
    total = 0
    for baseline, current in zip(baseline_predictions, current_predictions):
        changed += int(np.count_nonzero(baseline != current))
        total += int(baseline.size)
    return changed / max(total, 1)


def make_drift_dataset(records: list[ImageRecord], max_patches: int) -> EvalPatchDataset:
    max_per_image = max(1, math.ceil(max_patches / max(len(records), 1)))
    dataset = EvalPatchDataset(records, max_patches_per_image=max_per_image)
    if len(dataset.patch_records) > max_patches:
        dataset.patch_records = dataset.patch_records[:max_patches]
    return dataset


def load_gray_image(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    normalized = normalize_illumination(rgb)
    return cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)


def cached_load_gray(cache: OrderedDict[Path, np.ndarray], path: Path, *, max_cached_images: int) -> np.ndarray:
    image = cache.get(path)
    if image is not None:
        cache.move_to_end(path)
        return image

    image = load_gray_image(path)
    cache[path] = image
    cache.move_to_end(path)
    while len(cache) > max_cached_images:
        cache.popitem(last=False)
    return image


def normalize_illumination(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    low, high = np.percentile(lightness, (1, 99))
    if high > low:
        lightness = np.clip((lightness.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    merged = cv2.merge((lightness, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.35),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.5, p=0.9),
            A.RandomGamma(gamma_limit=(40, 200), p=0.6),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=5),
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                ],
                p=0.25,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(8, 28),
                hole_width_range=(8, 28),
                fill=0,
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        ]
    )


def sample_safe_rotated_patch(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    extraction = crop_random(image, TRAIN_MIN_EXTRACT, rng)
    angle = float(rng.uniform(*ROT_RANGE_DEG))
    scale = float(rng.uniform(*SCALE_RANGE))
    center = (extraction.shape[1] / 2.0, extraction.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        extraction,
        matrix,
        (extraction.shape[1], extraction.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return center_crop(rotated, PATCH_SIZE)


def crop_random(image: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    image = ensure_min_size(image, size)
    height, width = image.shape[:2]
    x0 = int(rng.integers(0, max(width - size, 0) + 1))
    y0 = int(rng.integers(0, max(height - size, 0) + 1))
    return crop_patch(image, x0, y0, size)


def crop_patch(image: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    image = ensure_min_size(image, size)
    patch = image[y0 : y0 + size, x0 : x0 + size]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = ensure_min_size(patch, size)
        patch = patch[:size, :size]
    return patch


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max((width - size) // 2, 0)
    y0 = max((height - size) // 2, 0)
    return crop_patch(image, x0, y0, size)


def ensure_min_size(image: np.ndarray, minimum: int) -> np.ndarray:
    height, width = image.shape[:2]
    pad_y = max(minimum - height, 0)
    pad_x = max(minimum - width, 0)
    if pad_y == 0 and pad_x == 0:
        return image
    return cv2.copyMakeBorder(
        image,
        pad_y // 2,
        pad_y - pad_y // 2,
        pad_x // 2,
        pad_x - pad_x // 2,
        borderType=cv2.BORDER_REFLECT_101,
    )


def preprocess_gray(image: np.ndarray) -> torch.Tensor:
    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor - GRAY_MEAN) / GRAY_STD
    return torch.from_numpy(tensor[None, :, :].copy())


def save_checkpoint(
    path: Path,
    model: SegformerClassificationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    split_records: dict[str, list[ImageRecord]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_type": "segformer_mit_b2_gray_classification",
            "model_state_dict": model.segmentation_model.state_dict(),
            "classification_head_state_dict": model.classification_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "history": history,
            "dataset_manifest_path": str(args.dataset_dir / "manifest.csv"),
            "training_config": json_safe_args(args),
            "preprocessing": {
                "patch_size": PATCH_SIZE,
                "eval_patch_stride": EVAL_PATCH_STRIDE,
                "gray_mean": GRAY_MEAN,
                "gray_std": GRAY_STD,
                "illumination_normalization": "LAB CLAHE L channel, percentile stretch 1/99, RGB to grayscale",
            },
            "split_counts": {
                split: {label: sum(record.label == label for record in records) for label in CLASS_NAMES}
                for split, records in split_records.items()
            },
        },
        path,
    )


def load_training_checkpoint(model: SegformerClassificationModel, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.segmentation_model.load_state_dict(checkpoint["model_state_dict"])
    model.classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
    model.to(device)


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    split_records: dict[str, list[ImageRecord]],
    device: torch.device,
) -> None:
    config = {
        "args": json_safe_args(args),
        "device": str(device),
        "model": "smp.Segformer encoder=mit_b2 in_channels=1 classes=4 with binary classification head",
        "class_names": CLASS_NAMES,
        "split_counts": {
            split: {label: sum(record.label == label for record in records) for label in CLASS_NAMES}
            for split, records in split_records.items()
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.inference_mode()
def generate_demo_outputs(
    model: SegformerClassificationModel,
    records: list[ImageRecord],
    payload: dict[str, np.ndarray],
    demo_dir: Path,
    device: torch.device,
) -> None:
    probabilities = payload["probabilities"]
    order = np.argsort(np.abs(probabilities - 0.5))[: min(6, len(records))]
    rows: list[dict[str, object]] = []

    for rank, index in enumerate(order, start=1):
        record = records[int(index)]
        gray = load_gray_image(record.image_path)
        patch = center_crop(ensure_min_size(gray, PATCH_SIZE), PATCH_SIZE)
        input_tensor = preprocess_gray(patch).unsqueeze(0).to(device)
        segmentation_logits, classification_logits = model.forward_segmentation_and_classification(input_tensor)
        segmentation = torch.argmax(segmentation_logits[0], dim=0).cpu().numpy().astype(np.uint8)
        tile_probability = float(torch.sigmoid(classification_logits[0]).detach().cpu().item())
        mask_rgb = SEGMENTATION_COLORS[segmentation]
        heatmap = probability_heatmap(tile_probability, PATCH_SIZE)

        output_path = demo_dir / f"demo_{rank:02d}_{record.label}_{record.sha256[:8]}.png"
        write_demo_figure(
            output_path,
            patch,
            mask_rgb,
            heatmap,
            true_label=record.label,
            image_probability=float(probabilities[index]),
            tile_probability=tile_probability,
        )
        rows.append(
            {
                "path": str(record.image_path),
                "true_label": record.label,
                "image_probability_difficult": float(probabilities[index]),
                "center_tile_probability_difficult": tile_probability,
                "demo_path": str(output_path),
            }
        )

    (demo_dir / "demo_index.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def probability_heatmap(probability: float, size: int) -> np.ndarray:
    canvas = np.full((size, size), probability, dtype=np.float32)
    return cv2.GaussianBlur(canvas, (0, 0), sigmaX=size / 8, sigmaY=size / 8)


def write_demo_figure(
    output_path: Path,
    gray: np.ndarray,
    mask_rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    true_label: str,
    image_probability: float,
    tile_probability: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input 512x512 gray tile")
    axes[1].imshow(mask_rgb)
    axes[1].set_title("SegFormer segmentation mask")
    image = axes[2].imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].text(
        0.5,
        0.5,
        f"{tile_probability:.2f}",
        transform=axes[2].transAxes,
        ha="center",
        va="center",
        fontsize=18,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    axes[2].set_title("Difficulty probability heatmap")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"true={true_label}; image p(difficult)={image_probability:.3f}; center tile p(difficult)={tile_probability:.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
