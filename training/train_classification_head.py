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

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


Image.MAX_IMAGE_PIXELS = None

NUM_SEGMENTATION_CLASSES = 4
CLASS_NAMES = ("ordinary", "difficult")
PATCH_SIZE = 256
EVAL_PATCH_STRIDE = 192
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ImageRecord:
    split: str
    label: str
    label_id: int
    sha256: str
    image_path: Path


@dataclass(frozen=True)
class PatchRecord:
    image_index: int
    x0: int
    y0: int


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int = 2048, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.linear(self.dropout(pooled)).squeeze(1)


class ClassificationModel(nn.Module):
    def __init__(self, segmentation_model: nn.Module, classification_head: nn.Module) -> None:
        super().__init__()
        self.segmentation_model = segmentation_model
        self.classification_head = classification_head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.segmentation_model.encoder(images)
        return self.classification_head(features[-1])


class RandomPatchDataset(Dataset):
    def __init__(self, records: list[ImageRecord], patches_per_image: int, seed: int) -> None:
        self.records = list(records)
        random.Random(seed).shuffle(self.records)
        self.patches_per_image = patches_per_image
        self.seed = seed
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index = index // self.patches_per_image
        record = self.records[image_index]
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        image = cached_load_image(self.image_cache, record.image_path, max_cached_images=4)
        image = sample_random_patch(image, rng)
        image = augment_patch(image, rng)
        return preprocess_image(image), torch.tensor(float(record.label_id), dtype=torch.float32)


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
        image = cached_load_image(self.image_cache, image_record.image_path, max_cached_images=4)
        patch = crop_patch(image, patch_record.x0, patch_record.y0)
        return (
            preprocess_image(patch),
            torch.tensor(float(image_record.label_id), dtype=torch.float32),
            torch.tensor(patch_record.image_index, dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ore difficulty classification head.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("training-artifacts/classification_dataset/v1"))
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training-artifacts/classification_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--head-only-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patches-per-image", type=int, default=32)
    parser.add_argument("--eval-max-patches-per-image", type=int, default=64)
    parser.add_argument("--drift-max-patches", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--drift-threshold", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Smoke-test limiter.")
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip training and evaluate an existing classification checkpoint on the test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_path is None:
        args.checkpoint_path = default_checkpoint_path()
    set_seed(args.seed)

    run_id = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    checkpoint_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(args.dataset_dir / "manifest.csv")
    if args.max_images_per_class is not None:
        records = limit_records(records, args.max_images_per_class)
    split_records = {split: [record for record in records if record.split == split] for split in ("train", "val", "test")}
    if any(not split_records[split] for split in split_records):
        raise RuntimeError(f"Empty split after loading {args.dataset_dir / 'manifest.csv'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_model = build_segmentation_model(args.checkpoint_path, device)
    classification_head = ClassificationHead().to(device)
    model = ClassificationModel(segmentation_model, classification_head).to(device)

    freeze_segmentation_decoder(segmentation_model)
    configure_trainable_parameters(segmentation_model, classification_head, train_layer4=False)
    optimizer = make_optimizer(classification_head, segmentation_model, args, train_layer4=False)

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
        write_test_report(
            run_dir,
            plots_dir,
            test_loss,
            test_metrics,
            test_payload,
            split_records["test"],
            checkpoint_path=args.evaluate_checkpoint,
        )
        print(f"Evaluated {args.evaluate_checkpoint}")
        print((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        return

    baseline_segmentation = collect_segmentation_predictions(segmentation_model, drift_loader, device)
    history: list[dict[str, float | int | str]] = []
    best_score: tuple[float, float] | None = None
    best_path: Path | None = None
    recommended_path: Path | None = None
    recommended_score: tuple[float, float] | None = None

    write_run_config(run_dir, args, split_records, device)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_only_epochs + 1:
            configure_trainable_parameters(segmentation_model, classification_head, train_layer4=True)
            optimizer = make_optimizer(classification_head, segmentation_model, args, train_layer4=True)

        train_layer4 = epoch > args.head_only_epochs
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, train_layer4=train_layer4)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, split_records["val"], device)
        drift = segmentation_drift(segmentation_model, drift_loader, baseline_segmentation, device)
        row = {
            "epoch": epoch,
            "phase": "head_only" if epoch <= args.head_only_epochs else "layer4_finetune",
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
            recommended_path = checkpoint_path
            save_checkpoint(checkpoint_dir / "recommended.pth", model, optimizer, epoch, history, args, split_records)

        write_history(run_dir / "history.csv", history)
        plot_history(history, plots_dir)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"drift={drift:.4f}"
        )

    if recommended_path is None and best_path is not None:
        save_checkpoint(checkpoint_dir / "recommended.pth", model, optimizer, int(history[-1]["epoch"]), history, args, split_records)

    recommended_checkpoint = checkpoint_dir / "recommended.pth"
    load_training_checkpoint(model, recommended_checkpoint, device)
    test_loss, test_metrics, test_payload = evaluate(
        model,
        test_loader,
        criterion,
        split_records["test"],
        device,
        return_payload=True,
    )
    test_report = write_test_report(
        run_dir,
        plots_dir,
        test_loss,
        test_metrics,
        test_payload,
        split_records["test"],
        checkpoint_path=recommended_checkpoint,
        best_path=best_path,
    )
    print(json.dumps(test_report, indent=2, ensure_ascii=False))
    print(f"Run artifacts written to {run_dir}")


def load_manifest(path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            records.append(
                ImageRecord(
                    split=row["split"],
                    label=row["label"],
                    label_id=int(row["label_id"]),
                    sha256=row["sha256"],
                    image_path=Path(row["dataset_path"]),
                )
            )
    return records


def default_checkpoint_path() -> Path:
    container_path = Path("/model-artifacts/ml-days-2/saves/epoch_014.pth")
    return container_path if container_path.exists() else Path("model-artifacts/ml-days-2/saves/epoch_014.pth")


def limit_records(records: list[ImageRecord], max_images_per_class: int) -> list[ImageRecord]:
    limited: list[ImageRecord] = []
    for split in ("train", "val", "test"):
        for label_id in (0, 1):
            subset = [record for record in records if record.split == split and record.label_id == label_id]
            limited.extend(subset[:max_images_per_class])
    return limited


def build_segmentation_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_SEGMENTATION_CLASSES,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    train_layer4: bool,
) -> None:
    for parameter in segmentation_model.encoder.parameters():
        parameter.requires_grad = False
    if train_layer4:
        for parameter in segmentation_model.encoder.layer4.parameters():
            parameter.requires_grad = True
    for parameter in classification_head.parameters():
        parameter.requires_grad = True


def make_optimizer(
    classification_head: nn.Module,
    segmentation_model: nn.Module,
    args: argparse.Namespace,
    *,
    train_layer4: bool,
) -> torch.optim.Optimizer:
    groups = [{"params": classification_head.parameters(), "lr": args.head_lr}]
    if train_layer4:
        groups.append({"params": segmentation_model.encoder.layer4.parameters(), "lr": args.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def train_epoch(
    model: ClassificationModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train_layer4: bool,
) -> float:
    model.train()
    model.segmentation_model.encoder.eval()
    if train_layer4:
        model.segmentation_model.encoder.layer4.train()
    model.segmentation_model.decoder.eval()
    model.segmentation_model.segmentation_head.eval()
    model.classification_head.train()
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
    model: ClassificationModel,
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


def binary_metrics(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    return metrics


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


def class_pos_weight(records: list[ImageRecord]) -> torch.Tensor:
    positives = sum(record.label_id for record in records)
    negatives = len(records) - positives
    return torch.tensor([negatives / max(positives, 1)], dtype=torch.float32)


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def cached_load_image(cache: OrderedDict[Path, np.ndarray], path: Path, *, max_cached_images: int) -> np.ndarray:
    image = cache.get(path)
    if image is not None:
        cache.move_to_end(path)
        return image

    image = load_image(path)
    cache[path] = image
    cache.move_to_end(path)
    while len(cache) > max_cached_images:
        cache.popitem(last=False)
    return image


def sample_random_patch(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    padded = ensure_min_size(image, PATCH_SIZE)
    height, width = padded.shape[:2]
    x0 = int(rng.integers(0, max(width - PATCH_SIZE, 0) + 1))
    y0 = int(rng.integers(0, max(height - PATCH_SIZE, 0) + 1))
    return crop_patch(padded, x0, y0)


def crop_patch(image: np.ndarray, x0: int, y0: int) -> np.ndarray:
    image = ensure_min_size(image, PATCH_SIZE)
    patch = image[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
    if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
        patch = ensure_min_size(patch, PATCH_SIZE)
        patch = patch[:PATCH_SIZE, :PATCH_SIZE]
    return patch


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


def augment_patch(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
    if rng.random() < 0.35:
        image = np.ascontiguousarray(image[::-1])
    if rng.random() < 0.5:
        image = np.ascontiguousarray(np.rot90(image, int(rng.integers(1, 4))))
    if rng.random() < 0.75:
        alpha = float(rng.uniform(0.8, 1.2))
        beta = float(rng.uniform(-20, 20))
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    if rng.random() < 0.25:
        noise = rng.normal(0, 8, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(tensor.transpose(2, 0, 1).copy())


def sliding_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    final_position = length - window
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def save_checkpoint(
    path: Path,
    model: ClassificationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float | int | str]],
    args: argparse.Namespace,
    split_records: dict[str, list[ImageRecord]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.segmentation_model.state_dict(),
            "classification_head_state_dict": model.classification_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "history": history,
            "dataset_manifest_path": str(args.dataset_dir / "manifest.csv"),
            "training_config": json_safe_args(args),
            "split_counts": {
                split: {label: sum(record.label == label for record in records) for label in CLASS_NAMES}
                for split, records in split_records.items()
            },
        },
        path,
    )


def load_training_checkpoint(model: ClassificationModel, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.segmentation_model.load_state_dict(checkpoint["model_state_dict"])
    model.classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
    model.to(device)


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_test_report(
    run_dir: Path,
    plots_dir: Path,
    test_loss: float,
    test_metrics: dict[str, float],
    test_payload: dict[str, np.ndarray],
    test_records: list[ImageRecord],
    *,
    checkpoint_path: Path,
    best_path: Path | None = None,
) -> dict[str, object]:
    test_report = {
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "evaluated_checkpoint": str(checkpoint_path),
        "best_val_auc_checkpoint": str(best_path) if best_path else None,
        "recommended_checkpoint": str(checkpoint_path),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(test_report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_test_outputs(test_payload, test_records, plots_dir)
    write_sample_predictions(test_payload, test_records, plots_dir)
    return test_report


def write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    split_records: dict[str, list[ImageRecord]],
    device: torch.device,
) -> None:
    config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "class_names": CLASS_NAMES,
        "split_counts": {
            split: {label: sum(record.label == label for record in records) for label in CLASS_NAMES}
            for split, records in split_records.items()
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def write_history(path: Path, history: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float | int | str]], plots_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], marker="o", label="train loss")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history], marker="o", label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, [float(row["val_accuracy"]) for row in history], marker="o", label="accuracy")
    axes[1].plot(epochs, [float(row["val_balanced_accuracy"]) for row in history], marker="o", label="balanced accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_accuracy.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key in ("val_precision", "val_recall", "val_f1", "val_roc_auc", "val_pr_auc"):
        axes[0].plot(epochs, [float(row[key]) for row in history], marker="o", label=key.replace("val_", ""))
    axes[0].set_title("Validation Classification Metrics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, [float(row["segmentation_drift"]) for row in history], marker="o", color="crimson")
    axes[1].set_title("Segmentation Drift vs Baseline")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Changed pixel ratio")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "classification_metrics.png", dpi=160)
    fig.savefig(plots_dir / "segmentation_drift.png", dpi=160)
    plt.close(fig)


def plot_test_outputs(payload: dict[str, np.ndarray], records: list[ImageRecord], plots_dir: Path) -> None:
    y_true = payload["y_true"]
    probabilities = payload["probabilities"]
    predictions = payload["predictions"]
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for y in range(2):
        for x in range(2):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix_test.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].plot(recall, precision)
    axes[0].set_title("Test ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Test Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_pr_test.png", dpi=160)
    plt.close(fig)

    rows = []
    for record, probability, prediction in zip(records, probabilities, predictions):
        rows.append(
            {
                "path": str(record.image_path),
                "label": record.label,
                "probability_difficult": float(probability),
                "prediction": CLASS_NAMES[int(prediction)],
            }
        )
    (plots_dir.parent / "test_predictions.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_sample_predictions(payload: dict[str, np.ndarray], records: list[ImageRecord], plots_dir: Path) -> None:
    probabilities = payload["probabilities"]
    predictions = payload["predictions"]
    order = np.argsort(np.abs(probabilities - 0.5))[:8]
    if len(order) == 0:
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, index in zip(axes.ravel(), order):
        record = records[int(index)]
        image = load_image(record.image_path)
        thumbnail = resize_for_display(image)
        axis.imshow(thumbnail)
        axis.set_title(
            f"true={record.label}\np(diff)={probabilities[index]:.3f}, pred={CLASS_NAMES[int(predictions[index])]}"
        )
        axis.axis("off")
    for axis in axes.ravel()[len(order) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "sample_predictions.png", dpi=160)
    plt.close(fig)


def resize_for_display(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    main()
