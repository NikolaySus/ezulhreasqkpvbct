from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_classification_head import ImageRecord, limit_records, load_manifest, seed_worker, set_seed, sigmoid
from training.train_global_context_segformer import (
    CLASSIFICATION_CLASS_NAMES,
    CLASS_NAMES,
    EVAL_PATCH_STRIDE,
    GLOBAL_SIZE,
    GRAY_MEAN,
    GRAY_STD,
    NUM_CLASSES,
    PATCH_SIZE,
    SEGMENTATION_COLORS,
    TRAIN_PATCHES_PER_IMAGE,
    CombinedSegmentationLoss,
    ContextAdapter,
    SegmentationRecord,
    binary_metrics,
    cached_load_gray,
    cached_load_mask,
    center_crop,
    class_pos_weight,
    classification_score,
    crop_patch,
    discover_segmentation_records,
    ensure_min_size,
    evaluate_segmentation,
    generate_demo_outputs,
    global_preprocess,
    global_roi_for_patch,
    json_safe_args,
    load_gray_image,
    preprocess_gray,
    sample_safe_rotated_pair_with_roi,
    sample_safe_rotated_patch_with_roi,
    sliding_positions,
    split_records,
    train_transforms,
    write_history,
    write_training_summary,
)


Image.MAX_IMAGE_PIXELS = None

CLASS_WEIGHTS = torch.tensor([1.0, 0.8, 0.6, 0.15], dtype=torch.float32)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(self.pool(features).flatten(1))).squeeze(1)


class SharedContextSegformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights=None,
            in_channels=1,
            classes=NUM_CLASSES,
        )
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head
        self.context_adapter = ContextAdapter(channels=512)
        self.classification_head = ClassificationHead(in_channels=512)
        self.pretrain_projection = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
        )
        self.style_projection = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 32, kernel_size=1),
        )

    def encode(self, images: torch.Tensor) -> list[torch.Tensor]:
        return list(self.encoder(images))

    def forward(
        self,
        local_images: torch.Tensor,
        global_images: torch.Tensor,
        global_rois: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_features = self.encode(local_images)
        global_features = self.encode(global_images)
        local_features[-1] = self.context_adapter(local_features[-1], global_features[-1], global_rois)
        decoder_output = self.decoder(local_features)
        segmentation_logits = self.segmentation_head(decoder_output)
        classification_logits = self.classification_head(local_features[-1])
        return segmentation_logits, classification_logits

    def classify(
        self,
        local_images: torch.Tensor,
        global_images: torch.Tensor,
        global_rois: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_features = self.encode(local_images)
        global_features = self.encode(global_images)
        local_features[-1] = self.context_adapter(local_features[-1], global_features[-1], global_rois)
        return self.classification_head(local_features[-1])

    def pretrain_features(
        self,
        local_images: torch.Tensor,
        local_images_second: torch.Tensor,
        global_images: torch.Tensor,
        global_rois: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local = self.encode(local_images)[-1]
        local_second = self.encode(local_images_second)[-1]
        global_feature = self.encode(global_images)[-1]
        modulated = self.context_adapter(local, global_feature, global_rois)
        return {
            "local": F.normalize(self.pretrain_projection(local), dim=1),
            "local_second": F.normalize(self.pretrain_projection(local_second), dim=1),
            "global": F.normalize(self.pretrain_projection(global_feature), dim=1),
            "style_local": self.style_projection(modulated),
            "style_global": self.style_projection(global_feature),
        }


class PretrainPatchDataset(Dataset):
    def __init__(self, records: list[SegmentationRecord], *, patches_per_image: int, seed: int, train: bool) -> None:
        self.records = list(records)
        random.Random(seed).shuffle(self.records)
        self.patches_per_image = patches_per_image
        self.train = train
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.mask_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.transforms = train_transforms() if train else None
        self.light_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.35),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.3, p=0.8),
                A.RandomGamma(gamma_limit=(60, 160), p=0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index // self.patches_per_image]
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        image = cached_load_gray(self.image_cache, record.image_path)
        mask = cached_load_mask(self.mask_cache, record.mask_path)
        global_image = global_preprocess(image)

        if self.train:
            local_image, local_mask, roi = sample_safe_rotated_pair_with_roi(image, mask, rng)
        else:
            x0 = int(rng.integers(0, max(record.width - PATCH_SIZE, 0) + 1))
            y0 = int(rng.integers(0, max(record.height - PATCH_SIZE, 0) + 1))
            local_image = crop_patch(image, x0, y0, PATCH_SIZE)
            local_mask = crop_patch(mask, x0, y0, PATCH_SIZE)
            roi = global_roi_for_patch(x0, y0, PATCH_SIZE, PATCH_SIZE, record.width, record.height)

        local_second = local_image.copy()
        if self.transforms is not None:
            augmented = self.transforms(image=local_image, mask=local_mask)
            local_image = augmented["image"]
            local_mask = augmented["mask"]
            local_second = self.light_transforms(image=local_second)["image"]

        return (
            preprocess_gray(local_image),
            preprocess_gray(local_second),
            global_image,
            torch.tensor(roi, dtype=torch.float32),
            torch.from_numpy(local_mask.astype(np.int64).copy()),
        )


class ClassificationPatch:
    def __init__(self, image_index: int, x0: int, y0: int) -> None:
        self.image_index = image_index
        self.x0 = x0
        self.y0 = y0


class ClassificationGlobalPatchDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        *,
        patches_per_image: int | None = None,
        max_patches_per_image: int | None = None,
        seed: int = 42,
        train: bool,
    ) -> None:
        self.records = list(records)
        self.train = train
        self.patches_per_image = patches_per_image
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.global_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self.transforms = train_transforms() if train else None
        self.patch_records: list[ClassificationPatch] = []
        if train:
            if patches_per_image is None:
                raise ValueError("patches_per_image is required for train classification dataset")
            random.Random(seed).shuffle(self.records)
        else:
            for image_index, record in enumerate(self.records):
                with Image.open(record.image_path) as image:
                    width, height = image.size
                patches = [
                    ClassificationPatch(image_index=image_index, x0=x0, y0=y0)
                    for y0 in sliding_positions(height, PATCH_SIZE, EVAL_PATCH_STRIDE)
                    for x0 in sliding_positions(width, PATCH_SIZE, EVAL_PATCH_STRIDE)
                ]
                if max_patches_per_image is not None and len(patches) > max_patches_per_image:
                    picks = np.linspace(0, len(patches) - 1, max_patches_per_image).round().astype(int)
                    patches = [patches[int(i)] for i in picks]
                self.patch_records.extend(patches)

    def __len__(self) -> int:
        if self.train:
            return len(self.records) * int(self.patches_per_image)
        return len(self.patch_records)

    def __getitem__(self, index: int):
        if self.train:
            record = self.records[index // int(self.patches_per_image)]
            rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
            image = cached_load_gray(self.image_cache, record.image_path)
            local, roi = sample_safe_rotated_patch_with_roi(image, rng)
            if self.transforms is not None:
                local = self.transforms(image=local)["image"]
            return (
                preprocess_gray(local),
                self.cached_global(record.image_path, image),
                torch.tensor(roi, dtype=torch.float32),
                torch.tensor(float(record.label_id), dtype=torch.float32),
            )

        patch = self.patch_records[index]
        record = self.records[patch.image_index]
        image = cached_load_gray(self.image_cache, record.image_path)
        local = crop_patch(image, patch.x0, patch.y0, PATCH_SIZE)
        roi = global_roi_for_patch(patch.x0, patch.y0, PATCH_SIZE, PATCH_SIZE, image.shape[1], image.shape[0])
        return (
            preprocess_gray(local),
            self.cached_global(record.image_path, image),
            torch.tensor(roi, dtype=torch.float32),
            torch.tensor(float(record.label_id), dtype=torch.float32),
            torch.tensor(patch.image_index, dtype=torch.long),
        )

    def cached_global(self, path: Path, image: np.ndarray, max_cached_images: int = 16) -> torch.Tensor:
        cached = self.global_cache.get(path)
        if cached is not None:
            self.global_cache.move_to_end(path)
            return cached
        tensor = global_preprocess(image)
        self.global_cache[path] = tensor
        self.global_cache.move_to_end(path)
        while len(self.global_cache) > max_cached_images:
            self.global_cache.popitem(last=False)
        return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared-encoder global-context SegFormer from random init.")
    parser.add_argument("--segmentation-data-root", type=Path, default=Path("training-artifacts/segmentation_dataset/all-1783214091-1"))
    parser.add_argument("--classification-dataset-dir", type=Path, default=Path("training-artifacts/classification_dataset/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("training-artifacts/shared_context_segformer_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--segmentation-epochs", type=int, default=100)
    parser.add_argument("--classification-epochs", type=int, default=80)
    parser.add_argument("--pretrain-patience", type=int, default=8)
    parser.add_argument("--segmentation-patience", type=int, default=10)
    parser.add_argument("--classification-patience", type=int, default=4)
    parser.add_argument("--pretrain-min-delta", type=float, default=0.001)
    parser.add_argument("--segmentation-min-delta", type=float, default=0.001)
    parser.add_argument("--classification-min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--auto-batch-candidates", type=str, default="16,12,8,4,2,1")
    parser.add_argument("--classification-auto-batch-size", action="store_true")
    parser.add_argument("--classification-auto-batch-candidates", type=str, default="64,48,32,24,16,12,8,4")
    parser.add_argument("--patches-per-image", type=int, default=TRAIN_PATCHES_PER_IMAGE)
    parser.add_argument("--eval-max-patches-per-image", type=int, default=32)
    parser.add_argument("--lr-pretrain", type=float, default=1e-4)
    parser.add_argument("--lr-adapter", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prototype-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pretrain-stage", action="store_true")
    parser.add_argument("--skip-segmentation-stage", action="store_true")
    parser.add_argument("--skip-classification-stage", action="store_true")
    parser.add_argument("--max-segmentation-images", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.pretrain_epochs = 1
        args.segmentation_epochs = 1
        args.classification_epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.patches_per_image = min(args.patches_per_image, 2)
        args.eval_max_patches_per_image = min(args.eval_max_patches_per_image, 2)
        args.max_images_per_class = args.max_images_per_class or 2
        args.max_segmentation_images = args.max_segmentation_images or 4

    set_seed(args.seed)
    run_id = args.run_name or time.strftime("shared_context_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    checkpoint_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    demo_dir = run_dir / "demo"
    for directory in (checkpoint_dir, plots_dir, demo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharedContextSegformer().to(device)
    history: list[dict[str, float | int | str]] = []
    if args.resume_checkpoint is not None:
        history = load_checkpoint(model, args.resume_checkpoint, device)
    existing_history_path = run_dir / "history.csv"
    if existing_history_path.exists():
        history = read_history(existing_history_path)

    segmentation_records = discover_segmentation_records(args.segmentation_data_root)
    if args.max_segmentation_images is not None:
        segmentation_records = segmentation_records[: args.max_segmentation_images]
    if not segmentation_records:
        raise RuntimeError(f"No segmentation records found under {args.segmentation_data_root}")
    train_records, val_records = split_records(segmentation_records, val_ratio=0.15, seed=args.seed)

    if args.auto_batch_size and not args.smoke:
        args.batch_size = select_batch_size(model, train_records, args, device)

    write_run_config(run_dir, args, device)

    if not args.skip_pretrain_stage:
        best_pretrain = train_pretrain_stage(model, train_records, val_records, args, history, checkpoint_dir, plots_dir, device)
        load_checkpoint(model, best_pretrain, device)
        print(f"Loaded best pretrain checkpoint for segmentation: {best_pretrain}")

    if not args.skip_segmentation_stage:
        best_segmentation = train_segmentation_stage(model, train_records, val_records, args, history, checkpoint_dir, plots_dir, device)
        load_checkpoint(model, best_segmentation, device)
        print(f"Loaded best segmentation checkpoint for classification: {best_segmentation}")

    if args.skip_classification_stage:
        print(f"Run artifacts written to {run_dir}")
        return

    classification_records = load_manifest(args.classification_dataset_dir / "manifest.csv")
    if args.max_images_per_class is not None:
        classification_records = limit_records(classification_records, args.max_images_per_class)
    split_classification = {
        split: [record for record in classification_records if record.split == split] for split in ("train", "val", "test")
    }
    if any(not records for records in split_classification.values()):
        raise RuntimeError("Classification train/val/test split is empty after loading manifest")
    if args.classification_auto_batch_size:
        args.batch_size = select_classification_batch_size(model, split_classification["train"], args, device)
        write_run_config(run_dir, args, device)
    train_classification_stage(model, split_classification, args, history, checkpoint_dir, plots_dir, demo_dir, device)
    print(f"Run artifacts written to {run_dir}")


def select_batch_size(
    model: SharedContextSegformer,
    records: list[SegmentationRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    candidates = [int(value.strip()) for value in args.auto_batch_candidates.split(",") if value.strip()]
    candidates = [value for value in candidates if value > 0]
    if args.batch_size not in candidates:
        candidates.append(args.batch_size)
    candidates = sorted(set(candidates), reverse=True)
    original = args.batch_size
    for candidate in candidates:
        args.batch_size = candidate
        loader = DataLoader(
            PretrainPatchDataset(
                records[: max(1, min(len(records), candidate))],
                patches_per_image=1,
                seed=args.seed,
                train=True,
            ),
            batch_size=candidate,
            shuffle=False,
            num_workers=0,
        )
        try:
            batch = next(iter(loader))
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
            pretrain_step(model, batch, optimizer, args, device, update_weights=False)
            del optimizer, batch, loader
            torch.cuda.empty_cache()
            print(f"Selected batch size {candidate}")
            return candidate
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                args.batch_size = original
                raise
            print(f"Batch size {candidate} failed with CUDA OOM, trying smaller.")
            optimizer = None
            torch.cuda.empty_cache()
    args.batch_size = original
    raise RuntimeError("No auto-batch candidate fits in GPU memory")


def select_classification_batch_size(
    model: SharedContextSegformer,
    records: list[ImageRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    candidates = [int(value.strip()) for value in args.classification_auto_batch_candidates.split(",") if value.strip()]
    candidates = sorted({value for value in candidates if value > 0}, reverse=True)
    original = args.batch_size
    configure_for_classification_stage(model)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(records).to(device))
    for candidate in candidates:
        args.batch_size = candidate
        loader = DataLoader(
            ClassificationGlobalPatchDataset(records[: max(1, min(len(records), candidate))], patches_per_image=1, seed=args.seed, train=True),
            batch_size=candidate,
            shuffle=False,
            num_workers=0,
        )
        try:
            optimizer = make_optimizer(model, args, stage="classification")
            batch = next(iter(loader))
            classification_memory_probe(model, batch, criterion, optimizer, device)
            del batch, loader, optimizer
            torch.cuda.empty_cache()
            print(f"Selected classification batch size {candidate}")
            return candidate
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                args.batch_size = original
                raise
            print(f"Classification batch size {candidate} failed with CUDA OOM, trying smaller.")
            torch.cuda.empty_cache()
    args.batch_size = original
    raise RuntimeError("No classification auto-batch candidate fits in GPU memory")


def train_pretrain_stage(
    model: SharedContextSegformer,
    train_records: list[SegmentationRecord],
    val_records: list[SegmentationRecord],
    args: argparse.Namespace,
    history: list[dict[str, float | int | str]],
    checkpoint_dir: Path,
    plots_dir: Path,
    device: torch.device,
) -> Path:
    configure_for_pretrain_stage(model)
    optimizer = make_optimizer(model, args, stage="pretrain")
    best_checkpoint_path = checkpoint_dir / "pretrain_best.pth"
    last_checkpoint_path = checkpoint_dir / "pretrain_last.pth"
    train_loader = DataLoader(
        PretrainPatchDataset(train_records, patches_per_image=args.patches_per_image, seed=args.seed, train=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        PretrainPatchDataset(val_records, patches_per_image=max(1, args.patches_per_image // 4), seed=args.seed + 1, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    previous_rows = [row for row in history if row["stage"] == "pretrain"]
    completed_epochs = max((int(row["epoch"]) for row in previous_rows), default=0)
    best_row = min(previous_rows, key=lambda row: float(row.get("val_loss", float("inf"))), default=None)
    best_loss = float(best_row.get("val_loss", float("inf"))) if best_row is not None else float("inf")
    best_epoch = int(best_row["epoch"]) if best_row is not None else 0
    epochs_without_improvement = 0
    stopped_reason = "max_epochs"
    for epoch in range(completed_epochs + 1, args.pretrain_epochs + 1):
        train_metrics = train_pretrain_epoch(model, train_loader, optimizer, args, device)
        val_metrics = evaluate_pretrain(model, val_loader, args, device)
        row = {
            "stage": "pretrain",
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            **{f"train_{key}": value for key, value in train_metrics.items() if key != "loss"},
            **{f"val_{key}": value for key, value in val_metrics.items() if key != "loss"},
        }
        history.append(row)
        if args.save_epoch_checkpoints:
            save_checkpoint(checkpoint_dir / f"pretrain_epoch_{epoch:03d}.pth", model, optimizer, epoch, history, args)
        save_checkpoint(last_checkpoint_path, model, optimizer, epoch, history, args)
        if val_metrics["loss"] <= best_loss - args.pretrain_min_delta:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, history, args)
        else:
            epochs_without_improvement += 1
        write_history(checkpoint_dir.parent / "history.csv", history)
        plot_combined_history(history, plots_dir)
        write_training_summary(
            checkpoint_dir.parent,
            {
                "pretrain": {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    "completed_epochs": epoch,
                    "max_epochs": args.pretrain_epochs,
                    "patience": args.pretrain_patience,
                    "min_delta": args.pretrain_min_delta,
                    "epochs_without_improvement": epochs_without_improvement,
                    "stopped_reason": stopped_reason,
                    "best_checkpoint": str(best_checkpoint_path),
                    "last_checkpoint": str(last_checkpoint_path),
                }
            },
        )
        print(
            f"Pretrain epoch {epoch:03d}/{args.pretrain_epochs}: "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"proto={val_metrics['prototype']:.4f} style={val_metrics['style']:.4f} dense={val_metrics['dense']:.4f} "
            f"no_improve={epochs_without_improvement}/{args.pretrain_patience}"
        )
        if epochs_without_improvement >= args.pretrain_patience:
            stopped_reason = "early_stopping"
            write_training_summary(
                checkpoint_dir.parent,
                {
                    "pretrain": {
                        "best_epoch": best_epoch,
                        "best_val_loss": best_loss,
                        "completed_epochs": epoch,
                        "max_epochs": args.pretrain_epochs,
                        "patience": args.pretrain_patience,
                        "min_delta": args.pretrain_min_delta,
                        "epochs_without_improvement": epochs_without_improvement,
                        "stopped_reason": stopped_reason,
                        "best_checkpoint": str(best_checkpoint_path),
                        "last_checkpoint": str(last_checkpoint_path),
                    }
                },
            )
            break
    if not best_checkpoint_path.exists():
        save_checkpoint(best_checkpoint_path, model, optimizer, completed_epochs, history, args)
    return best_checkpoint_path


def train_segmentation_stage(
    model: SharedContextSegformer,
    train_records: list[SegmentationRecord],
    val_records: list[SegmentationRecord],
    args: argparse.Namespace,
    history: list[dict[str, float | int | str]],
    checkpoint_dir: Path,
    plots_dir: Path,
    device: torch.device,
) -> Path:
    criterion = CombinedSegmentationLoss()
    best_checkpoint_path = checkpoint_dir / "segmentation_best.pth"
    last_checkpoint_path = checkpoint_dir / "segmentation_last.pth"
    train_loader = DataLoader(
        SegmentationDataset(train_records, patches_per_image=args.patches_per_image, seed=args.seed, train=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_records, patches_per_image=max(1, args.patches_per_image // 4), seed=args.seed + 1, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    previous_rows = [row for row in history if row["stage"] == "segmentation"]
    completed_epochs = max((int(row["epoch"]) for row in previous_rows), default=0)
    best_row = min(previous_rows, key=lambda row: float(row.get("val_loss", float("inf"))), default=None)
    best_loss = float(best_row.get("val_loss", float("inf"))) if best_row is not None else float("inf")
    best_miou = float(best_row.get("val_mean_iou", -1.0)) if best_row is not None else -1.0
    best_epoch = int(best_row["epoch"]) if best_row is not None else 0
    epochs_without_improvement = 0
    stopped_reason = "max_epochs"
    optimizer: torch.optim.Optimizer | None = None
    for epoch in range(completed_epochs + 1, args.segmentation_epochs + 1):
        configure_for_segmentation_stage(model, epoch)
        if optimizer is None or epoch in (9, 21):
            optimizer = make_optimizer(model, args, stage="segmentation")
        train_loss = train_segmentation_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_segmentation(model, val_loader, criterion, device)
        row = {
            "stage": "segmentation",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if args.save_epoch_checkpoints:
            save_checkpoint(checkpoint_dir / f"segmentation_epoch_{epoch:03d}.pth", model, optimizer, epoch, history, args)
        save_checkpoint(last_checkpoint_path, model, optimizer, epoch, history, args)
        if val_loss <= best_loss - args.segmentation_min_delta:
            best_loss = val_loss
            best_miou = val_metrics["mean_iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch, history, args)
        else:
            epochs_without_improvement += 1
        write_history(checkpoint_dir.parent / "history.csv", history)
        plot_combined_history(history, plots_dir)
        write_training_summary(
            checkpoint_dir.parent,
            {
                "segmentation": {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    "best_val_mean_iou": best_miou,
                    "completed_epochs": epoch,
                    "max_epochs": args.segmentation_epochs,
                    "patience": args.segmentation_patience,
                    "min_delta": args.segmentation_min_delta,
                    "epochs_without_improvement": epochs_without_improvement,
                    "stopped_reason": stopped_reason,
                    "best_checkpoint": str(best_checkpoint_path),
                    "last_checkpoint": str(last_checkpoint_path),
                }
            },
        )
        print(
            f"Seg epoch {epoch:03d}/{args.segmentation_epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_miou={val_metrics['mean_iou']:.4f} "
            f"best_val_loss={best_loss:.4f} best_miou={best_miou:.4f} "
            f"no_improve={epochs_without_improvement}/{args.segmentation_patience}"
        )
        if epochs_without_improvement >= args.segmentation_patience:
            stopped_reason = "early_stopping"
            write_training_summary(
                checkpoint_dir.parent,
                {
                    "segmentation": {
                        "best_epoch": best_epoch,
                        "best_val_loss": best_loss,
                        "best_val_mean_iou": best_miou,
                        "completed_epochs": epoch,
                        "max_epochs": args.segmentation_epochs,
                        "patience": args.segmentation_patience,
                        "min_delta": args.segmentation_min_delta,
                        "epochs_without_improvement": epochs_without_improvement,
                        "stopped_reason": stopped_reason,
                        "best_checkpoint": str(best_checkpoint_path),
                        "last_checkpoint": str(last_checkpoint_path),
                    }
                },
            )
            break
    if not best_checkpoint_path.exists():
        save_checkpoint(best_checkpoint_path, model, optimizer, completed_epochs, history, args)
    return best_checkpoint_path


def train_classification_stage(
    model: SharedContextSegformer,
    split_records: dict[str, list[ImageRecord]],
    args: argparse.Namespace,
    history: list[dict[str, float | int | str]],
    checkpoint_dir: Path,
    plots_dir: Path,
    demo_dir: Path,
    device: torch.device,
) -> None:
    configure_for_classification_stage(model)
    optimizer = make_optimizer(model, args, stage="classification")
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(split_records["train"]).to(device))
    recommended_checkpoint_path = checkpoint_dir / "recommended.pth"
    last_checkpoint_path = checkpoint_dir / "classification_last.pth"
    train_loader = DataLoader(
        ClassificationGlobalPatchDataset(split_records["train"], patches_per_image=args.patches_per_image, seed=args.seed, train=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        ClassificationGlobalPatchDataset(split_records["val"], max_patches_per_image=args.eval_max_patches_per_image, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ClassificationGlobalPatchDataset(split_records["test"], max_patches_per_image=args.eval_max_patches_per_image, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    previous_rows = [row for row in history if row["stage"] == "classification"]
    completed_epochs = max((int(row["epoch"]) for row in previous_rows), default=0)
    best_row = min(previous_rows, key=lambda row: float(row.get("val_loss", float("inf"))), default=None)
    best_loss = float(best_row.get("val_loss", float("inf"))) if best_row is not None else float("inf")
    best_score = classification_score(best_row) if best_row is not None else (-1.0, -1.0)
    best_epoch = int(best_row["epoch"]) if best_row is not None else 0
    early_stopping_loss = best_loss
    epochs_without_improvement = 0
    stopped_reason = "max_epochs"
    if previous_rows and not recommended_checkpoint_path.exists():
        save_checkpoint(recommended_checkpoint_path, model, optimizer, completed_epochs, history, args)
    for epoch in range(completed_epochs + 1, args.classification_epochs + 1):
        train_loss = train_classification_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _ = evaluate_classification(model, val_loader, criterion, split_records["val"], device)
        row = {
            "stage": "classification",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if args.save_epoch_checkpoints:
            save_checkpoint(checkpoint_dir / f"classification_epoch_{epoch:03d}.pth", model, optimizer, epoch, history, args)
        save_checkpoint(last_checkpoint_path, model, optimizer, epoch, history, args)
        if val_loss < best_loss:
            best_loss = val_loss
            best_score = (float(val_metrics["f1"]), float(val_metrics["roc_auc"]))
            best_epoch = epoch
            save_checkpoint(recommended_checkpoint_path, model, optimizer, epoch, history, args)
        if val_loss <= early_stopping_loss - args.classification_min_delta:
            early_stopping_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        write_history(checkpoint_dir.parent / "history.csv", history)
        plot_combined_history(history, plots_dir)
        write_training_summary(
            checkpoint_dir.parent,
            {
                "classification": {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_loss,
                    "best_val_f1": best_score[0],
                    "best_val_roc_auc": best_score[1],
                    "completed_epochs": epoch,
                    "max_epochs": args.classification_epochs,
                    "patience": args.classification_patience,
                    "min_delta": args.classification_min_delta,
                    "epochs_without_improvement": epochs_without_improvement,
                    "stopped_reason": stopped_reason,
                    "best_checkpoint": str(recommended_checkpoint_path),
                    "last_checkpoint": str(last_checkpoint_path),
                }
            },
        )
        print(
            f"Cls epoch {epoch:03d}/{args.classification_epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_metrics['roc_auc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"best_val_loss={best_loss:.4f} best_f1={best_score[0]:.4f} "
            f"no_improve={epochs_without_improvement}/{args.classification_patience}"
        )
        if epochs_without_improvement >= args.classification_patience:
            stopped_reason = "early_stopping"
            break
    load_checkpoint(model, recommended_checkpoint_path, device)
    test_loss, test_metrics, payload = evaluate_classification(model, test_loader, criterion, split_records["test"], device)
    report = {"test_loss": test_loss, "test_metrics": test_metrics, "checkpoint": str(recommended_checkpoint_path)}
    (checkpoint_dir.parent / "test_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_training_summary(
        checkpoint_dir.parent,
        {
            "classification": {
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "best_val_f1": best_score[0],
                "best_val_roc_auc": best_score[1],
                "completed_epochs": max((int(row["epoch"]) for row in history if row["stage"] == "classification"), default=0),
                "max_epochs": args.classification_epochs,
                "patience": args.classification_patience,
                "min_delta": args.classification_min_delta,
                "epochs_without_improvement": epochs_without_improvement,
                "stopped_reason": stopped_reason,
                "best_checkpoint": str(recommended_checkpoint_path),
                "last_checkpoint": str(last_checkpoint_path),
                "test_metrics": test_metrics,
            }
        },
    )
    generate_demo_outputs(model, split_records["test"], payload, demo_dir, device)
    print(json.dumps(report, indent=2, ensure_ascii=False))


class SegmentationDataset(Dataset):
    def __init__(self, records: list[SegmentationRecord], *, patches_per_image: int, seed: int, train: bool) -> None:
        self.records = list(records)
        random.Random(seed).shuffle(self.records)
        self.patches_per_image = patches_per_image
        self.train = train
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.mask_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.transforms = train_transforms() if train else None

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index // self.patches_per_image]
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        image = cached_load_gray(self.image_cache, record.image_path)
        mask = cached_load_mask(self.mask_cache, record.mask_path)
        if self.train:
            local_image, local_mask, roi = sample_safe_rotated_pair_with_roi(image, mask, rng)
            if self.transforms is not None:
                augmented = self.transforms(image=local_image, mask=local_mask)
                local_image = augmented["image"]
                local_mask = augmented["mask"]
        else:
            x0 = int(rng.integers(0, max(record.width - PATCH_SIZE, 0) + 1))
            y0 = int(rng.integers(0, max(record.height - PATCH_SIZE, 0) + 1))
            local_image = crop_patch(image, x0, y0, PATCH_SIZE)
            local_mask = crop_patch(mask, x0, y0, PATCH_SIZE)
            roi = global_roi_for_patch(x0, y0, PATCH_SIZE, PATCH_SIZE, record.width, record.height)
        return (
            preprocess_gray(local_image),
            global_preprocess(image),
            torch.tensor(roi, dtype=torch.float32),
            torch.from_numpy(local_mask.astype(np.int64).copy()),
        )


def train_pretrain_epoch(
    model: SharedContextSegformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "prototype": 0.0, "style": 0.0, "dense": 0.0}
    samples = 0
    for batch in loader:
        metrics = pretrain_step(model, batch, optimizer, args, device)
        batch_size = int(batch[0].size(0))
        for key, value in metrics.items():
            totals[key] += float(value) * batch_size
        samples += batch_size
    return {key: value / max(samples, 1) for key, value in totals.items()}


def pretrain_step(
    model: SharedContextSegformer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    *,
    update_weights: bool = True,
) -> dict[str, float]:
    local_images, local_second, global_images, global_rois, masks = batch
    local_images = local_images.to(device, non_blocking=True)
    local_second = local_second.to(device, non_blocking=True)
    global_images = global_images.to(device, non_blocking=True)
    global_rois = global_rois.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    features = model.pretrain_features(local_images, local_second, global_images, global_rois)
    proto_loss = class_prototype_loss(features["local"], masks, args.temperature)
    global_local_loss = global_local_class_consistency_loss(features["local"], features["global"], masks, global_rois)
    style_loss = style_field_loss(features["style_local"], features["style_global"], global_rois)
    dense_loss = dense_view_consistency_loss(features["local"], features["local_second"])
    loss = args.prototype_weight * (proto_loss + global_local_loss) + args.style_weight * style_loss + args.dense_weight * dense_loss
    loss.backward()
    if update_weights:
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.item()),
        "prototype": float((proto_loss + global_local_loss).item()),
        "style": float(style_loss.item()),
        "dense": float(dense_loss.item()),
    }


@torch.inference_mode()
def evaluate_pretrain(
    model: SharedContextSegformer,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "prototype": 0.0, "style": 0.0, "dense": 0.0}
    samples = 0
    for batch in loader:
        local_images, local_second, global_images, global_rois, masks = batch
        local_images = local_images.to(device, non_blocking=True)
        local_second = local_second.to(device, non_blocking=True)
        global_images = global_images.to(device, non_blocking=True)
        global_rois = global_rois.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        features = model.pretrain_features(local_images, local_second, global_images, global_rois)
        proto_loss = class_prototype_loss(features["local"], masks, args.temperature)
        global_local_loss = global_local_class_consistency_loss(features["local"], features["global"], masks, global_rois)
        style_loss = style_field_loss(features["style_local"], features["style_global"], global_rois)
        dense_loss = dense_view_consistency_loss(features["local"], features["local_second"])
        loss = args.prototype_weight * (proto_loss + global_local_loss) + args.style_weight * style_loss + args.dense_weight * dense_loss
        batch_size = int(local_images.size(0))
        totals["loss"] += float(loss.item()) * batch_size
        totals["prototype"] += float((proto_loss + global_local_loss).item()) * batch_size
        totals["style"] += float(style_loss.item()) * batch_size
        totals["dense"] += float(dense_loss.item()) * batch_size
        samples += batch_size
    return {key: value / max(samples, 1) for key, value in totals.items()}


def class_prototype_loss(features: torch.Tensor, masks: torch.Tensor, temperature: float) -> torch.Tensor:
    _, channels, height, width = features.shape
    labels = F.interpolate(masks[:, None].float(), size=(height, width), mode="nearest").squeeze(1).long()
    weights = CLASS_WEIGHTS.to(features.device)
    losses: list[torch.Tensor] = []
    for batch_index in range(features.size(0)):
        vectors: list[torch.Tensor] = []
        class_ids: list[int] = []
        for class_index in range(NUM_CLASSES):
            selected = labels[batch_index] == class_index
            if int(selected.sum()) < 2:
                continue
            prototype = features[batch_index, :, selected].mean(dim=1)
            vectors.append(F.normalize(prototype, dim=0))
            class_ids.append(class_index)
        if len(vectors) < 2:
            continue
        prototypes = torch.stack(vectors, dim=0)
        logits = prototypes @ prototypes.t() / temperature
        targets = torch.arange(len(vectors), device=features.device)
        class_weight = torch.tensor([float(weights[class_id]) for class_id in class_ids], device=features.device)
        losses.append(F.cross_entropy(logits, targets, weight=class_weight))
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def global_local_class_consistency_loss(
    local_features: torch.Tensor,
    global_features: torch.Tensor,
    masks: torch.Tensor,
    global_rois: torch.Tensor,
) -> torch.Tensor:
    _, _, height, width = local_features.shape
    labels = F.interpolate(masks[:, None].float(), size=(height, width), mode="nearest").squeeze(1).long()
    roi_summary = F.normalize(pool_global_rois(global_features, global_rois), dim=1)
    weights = CLASS_WEIGHTS.to(local_features.device)
    losses: list[torch.Tensor] = []
    for batch_index in range(local_features.size(0)):
        class_losses: list[torch.Tensor] = []
        class_weights: list[torch.Tensor] = []
        for class_index in range(NUM_CLASSES):
            selected = labels[batch_index] == class_index
            if int(selected.sum()) < 2:
                continue
            prototype = F.normalize(local_features[batch_index, :, selected].mean(dim=1), dim=0)
            class_losses.append(1.0 - torch.sum(prototype * roi_summary[batch_index]))
            class_weights.append(weights[class_index])
        if class_losses:
            stacked_losses = torch.stack(class_losses)
            stacked_weights = torch.stack(class_weights)
            losses.append((stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-6))
    if not losses:
        return local_features.new_tensor(0.0)
    return torch.stack(losses).mean()


def style_field_loss(local_style: torch.Tensor, global_style: torch.Tensor, global_rois: torch.Tensor) -> torch.Tensor:
    roi_style = pool_global_rois(global_style, global_rois)
    local_summary = F.adaptive_avg_pool2d(local_style, 1).flatten(1)
    summary_loss = F.mse_loss(local_summary, roi_style)
    smooth_x = torch.mean(torch.abs(global_style[:, :, :, 1:] - global_style[:, :, :, :-1]))
    smooth_y = torch.mean(torch.abs(global_style[:, :, 1:, :] - global_style[:, :, :-1, :]))
    return summary_loss + 0.05 * (smooth_x + smooth_y)


def dense_view_consistency_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_summary = F.normalize(F.adaptive_avg_pool2d(first, 1).flatten(1), dim=1)
    second_summary = F.normalize(F.adaptive_avg_pool2d(second, 1).flatten(1), dim=1)
    return 1.0 - (first_summary * second_summary).sum(dim=1).clamp(-1.0, 1.0).mean()


def pool_global_rois(global_feature: torch.Tensor, global_rois: torch.Tensor | None) -> torch.Tensor:
    if global_rois is None:
        return F.adaptive_avg_pool2d(global_feature, 1).flatten(1)
    _, channels, height, width = global_feature.shape
    pooled: list[torch.Tensor] = []
    rois = global_rois.detach().float().clamp(0.0, 1.0).cpu().numpy()
    for index, roi in enumerate(rois):
        x0 = int(math.floor(float(roi[0]) * width))
        y0 = int(math.floor(float(roi[1]) * height))
        x1 = int(math.ceil(float(roi[2]) * width))
        y1 = int(math.ceil(float(roi[3]) * height))
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(x0 + 1, min(width, x1))
        y1 = max(y0 + 1, min(height, y1))
        pooled.append(F.adaptive_avg_pool2d(global_feature[index : index + 1, :, y0:y1, x0:x1], 1).flatten(1))
    return torch.cat(pooled, dim=0) if pooled else global_feature.new_zeros((0, channels))


def train_segmentation_epoch(
    model: SharedContextSegformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    samples = 0
    for local_images, global_images, global_rois, masks in loader:
        local_images = local_images.to(device, non_blocking=True)
        global_images = global_images.to(device, non_blocking=True)
        global_rois = global_rois.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(local_images, global_images, global_rois)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * local_images.size(0)
        samples += local_images.size(0)
    return running / max(samples, 1)


def train_classification_epoch(
    model: SharedContextSegformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    running = 0.0
    samples = 0
    for batch in loader:
        loss = classification_step(model, batch, criterion, optimizer, device)
        running += loss * batch[0].size(0)
        samples += batch[0].size(0)
    return running / max(samples, 1)


def classification_step(
    model: SharedContextSegformer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.eval()
    model.classification_head.train()
    local_images, global_images, global_rois, labels = batch
    local_images = local_images.to(device, non_blocking=True)
    global_images = global_images.to(device, non_blocking=True)
    global_rois = global_rois.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model.classify(local_images, global_images, global_rois)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def classification_memory_probe(
    model: SharedContextSegformer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.eval()
    model.classification_head.train()
    local_images, global_images, global_rois, labels = batch
    local_images = local_images.to(device, non_blocking=True)
    global_images = global_images.to(device, non_blocking=True)
    global_rois = global_rois.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model.classify(local_images, global_images, global_rois)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)


@torch.inference_mode()
def evaluate_classification(
    model: SharedContextSegformer,
    loader: DataLoader,
    criterion: nn.Module,
    records: list[ImageRecord],
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    patch_logits: dict[int, list[float]] = {index: [] for index in range(len(records))}
    running = 0.0
    samples = 0
    for local_images, global_images, global_rois, labels, image_indices in loader:
        local_images = local_images.to(device, non_blocking=True)
        global_images = global_images.to(device, non_blocking=True)
        global_rois = global_rois.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.classify(local_images, global_images, global_rois)
        loss = criterion(logits, labels)
        running += float(loss.item()) * local_images.size(0)
        samples += local_images.size(0)
        for image_index, logit in zip(image_indices.tolist(), logits.detach().cpu().tolist()):
            patch_logits[image_index].append(float(logit))
    y_true = np.array([record.label_id for record in records], dtype=np.int64)
    logits = np.array([np.mean(patch_logits[index]) for index in range(len(records))], dtype=np.float32)
    probabilities = sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return (
        running / max(samples, 1),
        binary_metrics(y_true, predictions, probabilities),
        {"y_true": y_true, "probabilities": probabilities, "predictions": predictions},
    )


def configure_for_pretrain_stage(model: SharedContextSegformer) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.encoder, model.context_adapter, model.pretrain_projection, model.style_projection):
        for parameter in module.parameters():
            parameter.requires_grad = True


def configure_for_segmentation_stage(model: SharedContextSegformer, epoch: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.context_adapter, model.decoder, model.segmentation_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    names = ("patch_embed4", "block4", "norm4")
    if epoch > 8:
        names = names + ("patch_embed3", "block3", "norm3")
    if epoch > 20:
        names = names + ("patch_embed2", "block2", "norm2", "patch_embed1", "block1", "norm1")
    for name in names:
        module = getattr(model.encoder, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def configure_for_classification_stage(model: SharedContextSegformer) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classification_head.parameters():
        parameter.requires_grad = True


def make_optimizer(model: SharedContextSegformer, args: argparse.Namespace, *, stage: str) -> torch.optim.Optimizer:
    groups: list[dict[str, object]] = []
    encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    adapter_params = [parameter for parameter in model.context_adapter.parameters() if parameter.requires_grad]
    decoder_params = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    segmentation_params = [parameter for parameter in model.segmentation_head.parameters() if parameter.requires_grad]
    classification_params = [parameter for parameter in model.classification_head.parameters() if parameter.requires_grad]
    pretrain_params = [
        parameter
        for module in (model.pretrain_projection, model.style_projection)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.lr_pretrain if stage == "pretrain" else args.lr_encoder})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": args.lr_adapter})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": args.lr_adapter})
    if segmentation_params:
        groups.append({"params": segmentation_params, "lr": args.lr_adapter})
    if classification_params:
        groups.append({"params": classification_params, "lr": args.lr_head})
    if pretrain_params:
        groups.append({"params": pretrain_params, "lr": args.lr_pretrain})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def save_checkpoint(
    path: Path,
    model: SharedContextSegformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float | int | str]],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_type": "shared_context_segformer_mit_b2",
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "segmentation_head_state_dict": model.segmentation_head.state_dict(),
            "context_adapter_state_dict": model.context_adapter.state_dict(),
            "classification_head_state_dict": model.classification_head.state_dict(),
            "pretrain_projection_state_dict": model.pretrain_projection.state_dict(),
            "style_projection_state_dict": model.style_projection.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": CLASS_NAMES,
            "classification_class_names": CLASSIFICATION_CLASS_NAMES,
            "history": history,
            "training_config": json_safe_args(args),
            "preprocessing": {
                "patch_size": PATCH_SIZE,
                "global_size": GLOBAL_SIZE,
                "eval_patch_stride": EVAL_PATCH_STRIDE,
                "gray_mean": GRAY_MEAN,
                "gray_std": GRAY_STD,
                "illumination_normalization": "LAB CLAHE L channel, percentile stretch 1/99, RGB to grayscale",
            },
        },
        path,
    )


def load_checkpoint(model: SharedContextSegformer, path: Path, device: torch.device) -> list[dict[str, float | int | str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    model.segmentation_head.load_state_dict(checkpoint["segmentation_head_state_dict"])
    model.context_adapter.load_state_dict(checkpoint["context_adapter_state_dict"])
    model.classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
    if "pretrain_projection_state_dict" in checkpoint:
        model.pretrain_projection.load_state_dict(checkpoint["pretrain_projection_state_dict"])
    if "style_projection_state_dict" in checkpoint:
        model.style_projection.load_state_dict(checkpoint["style_projection_state_dict"])
    model.to(device)
    return list(checkpoint.get("history", []))


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def plot_combined_history(history: list[dict[str, float | int | str]], plots_dir: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for stage in ("pretrain", "segmentation", "classification"):
        rows = [row for row in history if row["stage"] == stage]
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        ax.plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", label=f"{stage} train")
        ax.plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label=f"{stage} val")
    ax.set_title("Shared-context training loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss.png", dpi=160)
    plt.close(fig)
    plot_pretrain_quality(history, plots_dir)
    plot_segmentation_quality(history, plots_dir)
    plot_classification_quality(history, plots_dir)


def plot_pretrain_quality(history: list[dict[str, float | int | str]], plots_dir: Path) -> None:
    rows = [row for row in history if row["stage"] == "pretrain"]
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in (("val_prototype", "Prototype"), ("val_style", "Style"), ("val_dense", "Dense consistency")):
        ax.plot(epochs, [float(row.get(key, 0.0)) for row in rows], marker="o", label=label)
    ax.set_title("Pretrain validation components")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "pretrain_quality.png", dpi=160)
    plt.close(fig)


def plot_segmentation_quality(history: list[dict[str, float | int | str]], plots_dir: Path) -> None:
    rows = [row for row in history if row["stage"] == "segmentation"]
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [float(row.get("val_mean_iou", 0.0)) for row in rows], marker="o", label="mean IoU")
    for class_index, class_name in enumerate(CLASS_NAMES):
        key = f"val_iou_class_{class_index}"
        if key in rows[0]:
            ax.plot(epochs, [float(row.get(key, 0.0)) for row in rows], marker=".", label=f"IoU {class_name}")
    ax.set_title("Segmentation validation quality")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "segmentation_quality.png", dpi=160)
    plt.close(fig)


def plot_classification_quality(history: list[dict[str, float | int | str]], plots_dir: Path) -> None:
    rows = [row for row in history if row["stage"] == "classification"]
    if not rows:
        return
    metrics = (
        ("val_roc_auc", "ROC AUC"),
        ("val_pr_auc", "PR AUC"),
        ("val_f1", "F1"),
        ("val_balanced_accuracy", "Balanced accuracy"),
        ("val_recall", "Recall"),
        ("val_precision", "Precision"),
    )
    epochs = [int(row["epoch"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in metrics:
        if key in rows[0]:
            ax.plot(epochs, [float(row.get(key, 0.0)) for row in rows], marker="o", label=label)
    ax.set_title("Classification validation quality")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "classification_quality.png", dpi=160)
    plt.close(fig)


def write_run_config(run_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    config = {
        "args": json_safe_args(args),
        "device": str(device),
        "model": "Shared-encoder SegFormer mit_b2 with global/local context adapter",
        "notes": "Experimental random-init pipeline; not connected to app/tasks.py or production compose.",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
