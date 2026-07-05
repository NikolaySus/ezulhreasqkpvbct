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
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.train_classification_head import ImageRecord, limit_records, load_manifest, seed_worker, set_seed, sigmoid


Image.MAX_IMAGE_PIXELS = None

NUM_CLASSES = 4
CLASS_NAMES = ("ore", "matrix", "talc", "damage")
CLASSIFICATION_CLASS_NAMES = ("ordinary", "difficult")
CLASS_ID_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
PATCH_SIZE = 512
GLOBAL_SIZE = 512
EVAL_PATCH_STRIDE = 192
TRAIN_PATCHES_PER_IMAGE = 16
GRAY_MEAN = 0.449
GRAY_STD = 0.226
ROT_RANGE_DEG = (-180.0, 180.0)
SCALE_RANGE = (0.85, 1.15)
TRAIN_MIN_EXTRACT = int(math.ceil(PATCH_SIZE * math.sqrt(2.0) / SCALE_RANGE[0]) + 4)
SEGMENTATION_COLORS = np.array(
    [
        [237, 28, 36],
        [255, 242, 0],
        [63, 72, 204],
        [136, 0, 21],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class SegmentationRecord:
    stem: str
    image_path: Path
    mask_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class ClassificationPatch:
    image_index: int
    x0: int
    y0: int


class ContextAdapter(nn.Module):
    def __init__(self, channels: int = 512, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels * 2),
        )

    def forward(
        self,
        local_feature: torch.Tensor,
        global_feature: torch.Tensor,
        global_rois: torch.Tensor | None,
    ) -> torch.Tensor:
        global_summary = F.adaptive_avg_pool2d(global_feature, 1).flatten(1)
        roi_summary = pool_global_rois(global_feature, global_rois)
        context = torch.cat([global_summary, roi_summary], dim=1)
        gamma_beta = self.net(context)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        return local_feature * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.linear(self.dropout(pooled)).squeeze(1)


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.cross_entropy(logits, masks) + self.dice_weight * self.dice(logits, masks)


class GlobalContextSegformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights=None,
            in_channels=1,
            classes=NUM_CLASSES,
        )
        self.global_model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights=None,
            in_channels=1,
            classes=NUM_CLASSES,
        )
        self.context_adapter = ContextAdapter(channels=512)
        self.classification_head = ClassificationHead(in_channels=512)

    def forward(
        self,
        local_images: torch.Tensor,
        global_images: torch.Tensor,
        global_rois: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_features = self.global_model.encoder(global_images)
        local_features = list(self.local_model.encoder(local_images))
        local_features[-1] = self.context_adapter(local_features[-1], global_features[-1], global_rois)
        decoder_output = self.local_model.decoder(local_features)
        segmentation_logits = self.local_model.segmentation_head(decoder_output)
        classification_logits = self.classification_head(local_features[-1])
        return segmentation_logits, classification_logits


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


class SegmentationPatchDataset(Dataset):
    def __init__(
        self,
        records: list[SegmentationRecord],
        *,
        patches_per_image: int,
        seed: int,
        train: bool,
    ) -> None:
        self.records = list(records)
        random.Random(seed).shuffle(self.records)
        self.patches_per_image = patches_per_image
        self.train = train
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.mask_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.transforms = train_transforms() if train else None

    def __len__(self) -> int:
        return len(self.records) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index // self.patches_per_image]
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        image = cached_load_gray(self.image_cache, record.image_path)
        mask = cached_load_mask(self.mask_cache, record.mask_path)
        global_image = global_preprocess(image)

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

        return preprocess_gray(local_image), global_image, torch.tensor(roi, dtype=torch.float32), torch.from_numpy(local_mask.astype(np.int64).copy())


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
                    patches = [patches[i] for i in picks]
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
                global_preprocess(image),
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
            global_preprocess(image),
            torch.tensor(roi, dtype=torch.float32),
            torch.tensor(float(record.label_id), dtype=torch.float32),
            torch.tensor(patch.image_index, dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experimental global-context SegFormer.")
    parser.add_argument("--segmentation-data-root", type=Path, default=Path("data3/approach2_default"))
    parser.add_argument("--classification-dataset-dir", type=Path, default=Path("training-artifacts/classification_dataset/v1"))
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("training-artifacts/global_context_segformer_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--segmentation-epochs", type=int, default=60)
    parser.add_argument("--classification-epochs", type=int, default=8)
    parser.add_argument("--segmentation-patience", type=int, default=12)
    parser.add_argument("--segmentation-min-delta", type=float, default=0.001)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=TRAIN_PATCHES_PER_IMAGE)
    parser.add_argument("--eval-max-patches-per-image", type=int, default=32)
    parser.add_argument("--lr-adapter", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-segmentation-stage", action="store_true")
    parser.add_argument("--skip-classification-stage", action="store_true")
    parser.add_argument("--max-segmentation-images", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_checkpoint is None:
        args.base_checkpoint = default_base_checkpoint_path()
    if args.smoke:
        args.segmentation_epochs = 1
        args.classification_epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.patches_per_image = min(args.patches_per_image, 2)
        args.eval_max_patches_per_image = min(args.eval_max_patches_per_image, 2)
        args.max_images_per_class = args.max_images_per_class or 2
        args.max_segmentation_images = args.max_segmentation_images or 4

    set_seed(args.seed)
    run_id = args.run_name or time.strftime("global_context_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    checkpoint_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    demo_dir = run_dir / "demo"
    for directory in (checkpoint_dir, plots_dir, demo_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalContextSegformer().to(device)
    load_base_checkpoint(model, args.base_checkpoint, device)

    history: list[dict[str, float | int | str]] = []
    if args.resume_checkpoint is not None:
        history = load_checkpoint(model, args.resume_checkpoint, device)
    existing_history_path = run_dir / "history.csv"
    if existing_history_path.exists():
        history = read_history(existing_history_path)
    write_run_config(run_dir, args, device)

    if not args.skip_segmentation_stage:
        segmentation_records = discover_segmentation_records(args.segmentation_data_root)
        if args.max_segmentation_images is not None:
            segmentation_records = segmentation_records[: args.max_segmentation_images]
        if not segmentation_records:
            raise RuntimeError(f"No segmentation records found under {args.segmentation_data_root}")
        train_records, val_records = split_records(segmentation_records, val_ratio=0.15, seed=args.seed)
        best_segmentation_checkpoint = train_segmentation_stage(
            model,
            train_records,
            val_records,
            args,
            history,
            checkpoint_dir,
            plots_dir,
            device,
        )
        load_checkpoint(model, best_segmentation_checkpoint, device)
        print(f"Loaded best segmentation checkpoint for classification: {best_segmentation_checkpoint}")
    else:
        print("Skipping segmentation stage; training classification head on base/global-context model.")

    if args.skip_classification_stage:
        print("Skipping classification stage.")
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
    train_classification_stage(model, split_classification, args, history, checkpoint_dir, plots_dir, demo_dir, device)
    print(f"Run artifacts written to {run_dir}")


def train_segmentation_stage(
    model: GlobalContextSegformer,
    train_records: list[SegmentationRecord],
    val_records: list[SegmentationRecord],
    args: argparse.Namespace,
    history: list[dict[str, float | int | str]],
    checkpoint_dir: Path,
    plots_dir: Path,
    device: torch.device,
) -> Path:
    configure_for_segmentation_stage(model)
    optimizer = make_optimizer(model, args, include_classification_head=False)
    criterion = CombinedSegmentationLoss()
    best_checkpoint_path = checkpoint_dir / "segmentation_best.pth"
    last_checkpoint_path = checkpoint_dir / "segmentation_last.pth"
    train_loader = DataLoader(
        SegmentationPatchDataset(train_records, patches_per_image=args.patches_per_image, seed=args.seed, train=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        SegmentationPatchDataset(val_records, patches_per_image=max(1, args.patches_per_image // 4), seed=args.seed + 1, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    previous_rows = [row for row in history if row["stage"] == "segmentation"]
    completed_epochs = max((int(row["epoch"]) for row in previous_rows), default=0)
    best_row = max(previous_rows, key=lambda row: float(row.get("val_mean_iou", -1.0)), default=None)
    best_miou = float(best_row.get("val_mean_iou", -1.0)) if best_row is not None else -1.0
    best_epoch = int(best_row["epoch"]) if best_row is not None else 0
    epochs_without_improvement = 0
    stopped_reason = "max_epochs"
    if completed_epochs > 0:
        if not best_checkpoint_path.exists():
            save_checkpoint(best_checkpoint_path, model, optimizer, completed_epochs, history, args)
        save_checkpoint(last_checkpoint_path, model, optimizer, completed_epochs, history, args)
    if completed_epochs >= args.segmentation_epochs:
        print(
            f"Segmentation already has {completed_epochs} epochs; "
            f"target is {args.segmentation_epochs}, skipping training."
        )
    for epoch in range(completed_epochs + 1, args.segmentation_epochs + 1):
        train_loss = train_segmentation_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_segmentation(model, val_loader, criterion, device)
        row = {
            "stage": "segmentation",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if args.save_epoch_checkpoints:
            save_checkpoint(checkpoint_dir / f"segmentation_epoch_{epoch:03d}.pth", model, optimizer, epoch, history, args)
        save_checkpoint(last_checkpoint_path, model, optimizer, epoch, history, args)
        if val_metrics["mean_iou"] >= best_miou + args.segmentation_min_delta:
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
            f"Seg epoch {epoch:03d}/{args.segmentation_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_miou={val_metrics['mean_iou']:.4f} best_miou={best_miou:.4f} "
            f"no_improve={epochs_without_improvement}/{args.segmentation_patience}"
        )
        if epochs_without_improvement >= args.segmentation_patience:
            stopped_reason = "early_stopping"
            write_training_summary(
                checkpoint_dir.parent,
                {
                    "segmentation": {
                        "best_epoch": best_epoch,
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
                f"Segmentation early stopping at epoch {epoch}; "
                f"best epoch {best_epoch}, best val_miou={best_miou:.4f}"
            )
            break
    if not best_checkpoint_path.exists():
        save_checkpoint(best_checkpoint_path, model, optimizer, completed_epochs, history, args)
    return best_checkpoint_path


def train_classification_stage(
    model: GlobalContextSegformer,
    split_records: dict[str, list[ImageRecord]],
    args: argparse.Namespace,
    history: list[dict[str, float | int | str]],
    checkpoint_dir: Path,
    plots_dir: Path,
    demo_dir: Path,
    device: torch.device,
) -> None:
    configure_for_classification_stage(model)
    optimizer = make_optimizer(model, args, include_classification_head=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(split_records["train"]).to(device))
    train_loader = DataLoader(
        ClassificationGlobalPatchDataset(
            split_records["train"],
            patches_per_image=args.patches_per_image,
            seed=args.seed,
            train=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        ClassificationGlobalPatchDataset(
            split_records["val"],
            max_patches_per_image=args.eval_max_patches_per_image,
            train=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ClassificationGlobalPatchDataset(
            split_records["test"],
            max_patches_per_image=args.eval_max_patches_per_image,
            train=False,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    previous_rows = [row for row in history if row["stage"] == "classification"]
    completed_epochs = max((int(row["epoch"]) for row in previous_rows), default=0)
    best_auc = max((float(row.get("val_roc_auc", -1.0)) for row in previous_rows), default=-1.0)
    if completed_epochs >= args.classification_epochs:
        print(
            f"Classification already has {completed_epochs} epochs; "
            f"target is {args.classification_epochs}, skipping training."
        )
    for epoch in range(completed_epochs + 1, args.classification_epochs + 1):
        train_loss = train_classification_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, _ = evaluate_classification(model, val_loader, criterion, split_records["val"], device)
        row = {
            "stage": "classification",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if args.save_epoch_checkpoints:
            save_checkpoint(checkpoint_dir / f"classification_epoch_{epoch:03d}.pth", model, optimizer, epoch, history, args)
        if val_metrics["roc_auc"] >= best_auc:
            best_auc = val_metrics["roc_auc"]
            save_checkpoint(checkpoint_dir / "recommended.pth", model, optimizer, epoch, history, args)
        write_history(checkpoint_dir.parent / "history.csv", history)
        plot_combined_history(history, plots_dir)
        print(
            f"Cls epoch {epoch:03d}/{args.classification_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

    load_checkpoint(model, checkpoint_dir / "recommended.pth", device)
    test_loss, test_metrics, payload = evaluate_classification(
        model,
        test_loader,
        criterion,
        split_records["test"],
        device,
    )
    report = {"test_loss": test_loss, "test_metrics": test_metrics, "checkpoint": str(checkpoint_dir / "recommended.pth")}
    (checkpoint_dir.parent / "test_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_training_summary(
        checkpoint_dir.parent,
        {
            "classification": {
                "completed_epochs": args.classification_epochs,
                "best_checkpoint": str(checkpoint_dir / "recommended.pth"),
                "test_metrics": test_metrics,
            }
        },
    )
    generate_demo_outputs(model, split_records["test"], payload, demo_dir, device)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def train_segmentation_epoch(
    model: GlobalContextSegformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    model.global_model.eval()
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
    model: GlobalContextSegformer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.eval()
    model.context_adapter.train()
    model.classification_head.train()
    running = 0.0
    samples = 0
    for local_images, global_images, global_rois, labels in loader:
        local_images = local_images.to(device, non_blocking=True)
        global_images = global_images.to(device, non_blocking=True)
        global_rois = global_rois.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(local_images, global_images, global_rois)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * local_images.size(0)
        samples += local_images.size(0)
    return running / max(samples, 1)


@torch.inference_mode()
def evaluate_segmentation(
    model: GlobalContextSegformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    running = 0.0
    samples = 0
    intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    unions = np.zeros(NUM_CLASSES, dtype=np.float64)
    for local_images, global_images, global_rois, masks in loader:
        local_images = local_images.to(device, non_blocking=True)
        global_images = global_images.to(device, non_blocking=True)
        global_rois = global_rois.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits, _ = model(local_images, global_images, global_rois)
        loss = criterion(logits, masks)
        running += float(loss.item()) * local_images.size(0)
        samples += local_images.size(0)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        for class_index in range(NUM_CLASSES):
            pred_mask = predictions == class_index
            target_mask = targets == class_index
            intersections[class_index] += np.logical_and(pred_mask, target_mask).sum()
            unions[class_index] += np.logical_or(pred_mask, target_mask).sum()
    ious = intersections / np.maximum(unions, 1.0)
    metrics = {f"iou_class_{idx}": float(value) for idx, value in enumerate(ious)}
    metrics["mean_iou"] = float(np.mean(ious))
    return running / max(samples, 1), metrics


@torch.inference_mode()
def evaluate_classification(
    model: GlobalContextSegformer,
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
        _, logits = model(local_images, global_images, global_rois)
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


def configure_for_segmentation_stage(model: GlobalContextSegformer) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.context_adapter, model.local_model.decoder, model.local_model.segmentation_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    for name in ("patch_embed4", "block4", "norm4"):
        module = getattr(model.local_model.encoder, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def configure_for_classification_stage(model: GlobalContextSegformer) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.context_adapter, model.classification_head):
        for parameter in module.parameters():
            parameter.requires_grad = True


def make_optimizer(
    model: GlobalContextSegformer,
    args: argparse.Namespace,
    *,
    include_classification_head: bool,
) -> torch.optim.Optimizer:
    groups: list[dict[str, object]] = [{"params": model.context_adapter.parameters(), "lr": args.lr_adapter}]
    trainable_encoder_params = [p for p in model.local_model.encoder.parameters() if p.requires_grad]
    trainable_decoder_params = [p for p in model.local_model.decoder.parameters() if p.requires_grad]
    trainable_segmentation_head_params = [p for p in model.local_model.segmentation_head.parameters() if p.requires_grad]
    if trainable_encoder_params:
        groups.append({"params": trainable_encoder_params, "lr": args.lr_encoder})
    if trainable_decoder_params:
        groups.append({"params": trainable_decoder_params, "lr": args.lr_adapter})
    if trainable_segmentation_head_params:
        groups.append({"params": trainable_segmentation_head_params, "lr": args.lr_adapter})
    if include_classification_head:
        groups.append({"params": model.classification_head.parameters(), "lr": args.lr_head})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def discover_segmentation_records(root: Path) -> list[SegmentationRecord]:
    image_dir = root / "corrected-originals"
    mask_dir = root / "corrected-masks"
    records: list[SegmentationRecord] = []
    if not image_dir.exists() or not mask_dir.exists():
        return records
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        mask_path = mask_dir / f"{image_path.stem}.npy"
        if not mask_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(SegmentationRecord(image_path.stem, image_path, mask_path, width, height))
    return records


def split_records(records: list[SegmentationRecord], *, val_ratio: float, seed: int) -> tuple[list[SegmentationRecord], list[SegmentationRecord]]:
    records = list(records)
    random.Random(seed).shuffle(records)
    val_count = max(1, int(round(len(records) * val_ratio))) if len(records) > 1 else 0
    return records[val_count:], records[:val_count] if val_count else records


def default_base_checkpoint_path() -> Path:
    container_path = Path("/model-artifacts/ml-days-2/segformer/epoch_028.pth")
    return container_path if container_path.exists() else Path("model-artifacts/ml-days-2/segformer/epoch_028.pth")


def load_base_checkpoint(model: GlobalContextSegformer, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    model.local_model.load_state_dict(state)
    model.global_model.load_state_dict(state)


def save_checkpoint(
    path: Path,
    model: GlobalContextSegformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, float | int | str]],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_type": "global_context_segformer_mit_b2",
            "local_model_state_dict": model.local_model.state_dict(),
            "global_model_state_dict": model.global_model.state_dict(),
            "context_adapter_state_dict": model.context_adapter.state_dict(),
            "classification_head_state_dict": model.classification_head.state_dict(),
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


def load_checkpoint(model: GlobalContextSegformer, path: Path, device: torch.device) -> list[dict[str, float | int | str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.local_model.load_state_dict(checkpoint["local_model_state_dict"])
    model.global_model.load_state_dict(checkpoint["global_model_state_dict"])
    model.context_adapter.load_state_dict(checkpoint["context_adapter_state_dict"])
    model.classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
    model.to(device)
    return list(checkpoint.get("history", []))


def load_gray_image(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return cv2.cvtColor(normalize_illumination(rgb), cv2.COLOR_RGB2GRAY)


def cached_load_gray(cache: OrderedDict[Path, np.ndarray], path: Path, max_cached_images: int = 4) -> np.ndarray:
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


def cached_load_mask(cache: OrderedDict[Path, np.ndarray], path: Path, max_cached_masks: int = 4) -> np.ndarray:
    mask = cache.get(path)
    if mask is not None:
        cache.move_to_end(path)
        return mask
    mask = remap_mask(np.load(path).astype(np.uint8))
    cache[path] = mask
    cache.move_to_end(path)
    while len(cache) > max_cached_masks:
        cache.popitem(last=False)
    return mask


def remap_mask(mask: np.ndarray) -> np.ndarray:
    remapped = np.empty(mask.shape, dtype=np.uint8)
    for source_label, target_label in CLASS_ID_MAP.items():
        remapped[mask == source_label] = target_label
    return remapped


def normalize_illumination(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    low, high = np.percentile(lightness, (1, 99))
    if high > low:
        lightness = np.clip((lightness.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.35),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.5, p=0.9),
            A.RandomGamma(gamma_limit=(40, 200), p=0.6),
            A.OneOf([A.GaussianBlur(blur_limit=5), A.MotionBlur(blur_limit=5), A.MedianBlur(blur_limit=5)], p=0.25),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(8, 28), hole_width_range=(8, 28), fill=0, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        ]
    )


def sample_safe_rotated_pair_with_roi(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    original_height, original_width = image.shape[:2]
    image, mask = ensure_min_size_pair(image, mask, TRAIN_MIN_EXTRACT)
    height, width = image.shape[:2]
    x0 = int(rng.integers(0, max(width - TRAIN_MIN_EXTRACT, 0) + 1))
    y0 = int(rng.integers(0, max(height - TRAIN_MIN_EXTRACT, 0) + 1))
    local_image = image[y0 : y0 + TRAIN_MIN_EXTRACT, x0 : x0 + TRAIN_MIN_EXTRACT]
    local_mask = mask[y0 : y0 + TRAIN_MIN_EXTRACT, x0 : x0 + TRAIN_MIN_EXTRACT]
    angle = float(rng.uniform(*ROT_RANGE_DEG))
    scale = float(rng.uniform(*SCALE_RANGE))
    center = (TRAIN_MIN_EXTRACT / 2.0, TRAIN_MIN_EXTRACT / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(local_image, matrix, (TRAIN_MIN_EXTRACT, TRAIN_MIN_EXTRACT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    rotated_mask = cv2.warpAffine(local_mask, matrix, (TRAIN_MIN_EXTRACT, TRAIN_MIN_EXTRACT), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    offset = max((TRAIN_MIN_EXTRACT - PATCH_SIZE) // 2, 0)
    roi = global_roi_for_patch(
        x0 + offset,
        y0 + offset,
        PATCH_SIZE,
        PATCH_SIZE,
        max(original_width, width),
        max(original_height, height),
    )
    return center_crop(rotated_image, PATCH_SIZE), center_crop(rotated_mask, PATCH_SIZE), roi


def sample_safe_rotated_patch_with_roi(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    original_height, original_width = image.shape[:2]
    image = ensure_min_size(image, TRAIN_MIN_EXTRACT)
    height, width = image.shape[:2]
    x0 = int(rng.integers(0, max(width - TRAIN_MIN_EXTRACT, 0) + 1))
    y0 = int(rng.integers(0, max(height - TRAIN_MIN_EXTRACT, 0) + 1))
    local = image[y0 : y0 + TRAIN_MIN_EXTRACT, x0 : x0 + TRAIN_MIN_EXTRACT]
    angle = float(rng.uniform(*ROT_RANGE_DEG))
    scale = float(rng.uniform(*SCALE_RANGE))
    center = (TRAIN_MIN_EXTRACT / 2.0, TRAIN_MIN_EXTRACT / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(local, matrix, (TRAIN_MIN_EXTRACT, TRAIN_MIN_EXTRACT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    offset = max((TRAIN_MIN_EXTRACT - PATCH_SIZE) // 2, 0)
    roi = global_roi_for_patch(
        x0 + offset,
        y0 + offset,
        PATCH_SIZE,
        PATCH_SIZE,
        max(original_width, width),
        max(original_height, height),
    )
    return center_crop(rotated, PATCH_SIZE), roi


def global_roi_for_patch(
    x0: int,
    y0: int,
    patch_width: int,
    patch_height: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    scale = GLOBAL_SIZE / max(image_height, image_width)
    resized_width = image_width * scale
    resized_height = image_height * scale
    offset_x = (GLOBAL_SIZE - resized_width) / 2.0
    offset_y = (GLOBAL_SIZE - resized_height) / 2.0
    gx0 = (offset_x + x0 * scale) / GLOBAL_SIZE
    gy0 = (offset_y + y0 * scale) / GLOBAL_SIZE
    gx1 = (offset_x + (x0 + patch_width) * scale) / GLOBAL_SIZE
    gy1 = (offset_y + (y0 + patch_height) * scale) / GLOBAL_SIZE
    return (
        float(np.clip(gx0, 0.0, 1.0)),
        float(np.clip(gy0, 0.0, 1.0)),
        float(np.clip(gx1, 0.0, 1.0)),
        float(np.clip(gy1, 0.0, 1.0)),
    )


def global_preprocess(image: np.ndarray) -> torch.Tensor:
    height, width = image.shape[:2]
    scale = GLOBAL_SIZE / max(height, width)
    resized = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((GLOBAL_SIZE, GLOBAL_SIZE), dtype=np.uint8)
    y0 = (GLOBAL_SIZE - resized.shape[0]) // 2
    x0 = (GLOBAL_SIZE - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return preprocess_gray(canvas)


def preprocess_gray(image: np.ndarray) -> torch.Tensor:
    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor - GRAY_MEAN) / GRAY_STD
    return torch.from_numpy(tensor[None, :, :].copy())


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


def ensure_min_size_pair(image: np.ndarray, mask: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    pad_y = max(minimum - height, 0)
    pad_x = max(minimum - width, 0)
    if pad_y == 0 and pad_x == 0:
        return image, mask
    border = (pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2)
    image = cv2.copyMakeBorder(image, *border, borderType=cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, *border, borderType=cv2.BORDER_REFLECT_101)
    return image, mask


def sliding_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    final_position = length - window
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def class_pos_weight(records: list[ImageRecord]) -> torch.Tensor:
    positives = sum(record.label_id for record in records)
    negatives = len(records) - positives
    return torch.tensor([negatives / max(positives, 1)], dtype=torch.float32)


def binary_metrics(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average="binary", zero_division=0)
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


def write_history(path: Path, history: list[dict[str, float | int | str]]) -> None:
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_training_summary(run_dir: Path, update: dict[str, object]) -> None:
    path = run_dir / "training_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary.update(update)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_combined_history(history: list[dict[str, float | int | str]], plots_dir: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for stage in ("segmentation", "classification"):
        rows = [row for row in history if row["stage"] == stage]
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        ax.plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", label=f"{stage} train")
        ax.plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label=f"{stage} val")
    ax.set_title("Global-context training loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss.png", dpi=160)
    plt.close(fig)
    plot_segmentation_quality(history, plots_dir)
    plot_classification_quality(history, plots_dir)


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


@torch.inference_mode()
def generate_demo_outputs(
    model: GlobalContextSegformer,
    records: list[ImageRecord],
    payload: dict[str, np.ndarray],
    demo_dir: Path,
    device: torch.device,
) -> None:
    probabilities = payload["probabilities"]
    order = np.argsort(np.abs(probabilities - 0.5))[: min(6, len(records))]
    rows: list[dict[str, object]] = []
    model.eval()
    for rank, index in enumerate(order, start=1):
        record = records[int(index)]
        image = load_gray_image(record.image_path)
        local = center_crop(ensure_min_size(image, PATCH_SIZE), PATCH_SIZE)
        local_tensor = preprocess_gray(local).unsqueeze(0).to(device)
        global_tensor = global_preprocess(image).unsqueeze(0).to(device)
        segmentation_logits, classification_logits = model(local_tensor, global_tensor)
        mask = torch.argmax(segmentation_logits[0], dim=0).cpu().numpy().astype(np.uint8)
        tile_probability = float(torch.sigmoid(classification_logits[0]).cpu().item())
        output_path = demo_dir / f"demo_{rank:02d}_{record.label}_{record.sha256[:8]}.png"
        write_demo_figure(output_path, local, SEGMENTATION_COLORS[mask], tile_probability, record.label, float(probabilities[index]))
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


def write_demo_figure(
    output_path: Path,
    gray: np.ndarray,
    mask_rgb: np.ndarray,
    tile_probability: float,
    true_label: str,
    image_probability: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Local tile")
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Global-context segmentation")
    heatmap = np.full((PATCH_SIZE, PATCH_SIZE), tile_probability, dtype=np.float32)
    image = axes[2].imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].text(0.5, 0.5, f"{tile_probability:.2f}", transform=axes[2].transAxes, ha="center", va="center", fontsize=18, color="white")
    axes[2].set_title("Difficulty probability")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"true={true_label}; image p(difficult)={image_probability:.3f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_run_config(run_dir: Path, args: argparse.Namespace, device: torch.device) -> None:
    config = {
        "args": json_safe_args(args),
        "device": str(device),
        "model": "Global-local SegFormer mit_b2 with FiLM context adapter and binary classification head",
        "notes": "Experimental only; not connected to app/tasks.py or production compose.",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


if __name__ == "__main__":
    main()
