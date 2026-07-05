from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
LABEL_TO_ID = {"ordinary": 0, "difficult": 1}


@dataclass(frozen=True)
class SourceImage:
    label: str
    label_id: int
    source_path: Path
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build binary ore-classification dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training-artifacts/classification_dataset/v1"),
        help="Ignored output dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root containing the two 'Фото руд по сортам...' directories. Defaults to host path or /source-nornik.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true", help="Remove output dir before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if round(args.train_ratio + args.val_ratio + args.test_ratio, 6) != 1:
        raise ValueError("train/val/test ratios must sum to 1.0")

    sources = default_sources(args.source_root)
    records = discover_source_images(sources)
    unique_records, conflicts, duplicate_counts = deduplicate(records)
    train_records, val_records, test_records = split_records(
        unique_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dataset(args.output_dir, train_records, val_records, test_records)
    write_conflicts(args.output_dir / "conflicts.csv", conflicts)
    write_summary(
        args.output_dir / "split_summary.json",
        source_records=records,
        unique_records=unique_records,
        conflicts=conflicts,
        duplicate_counts=duplicate_counts,
        split_records={"train": train_records, "val": val_records, "test": test_records},
        args=args,
        sources=sources,
    )

    print(f"Source images: {len(records)}")
    print(f"Unique usable images: {len(unique_records)}")
    print(f"Conflicting SHA256 groups removed: {len(conflicts)}")
    for split_name, split in (("train", train_records), ("val", val_records), ("test", test_records)):
        counts = Counter(record.label for record in split)
        print(f"{split_name}: {len(split)} {dict(counts)}")
    print(f"Dataset written to {args.output_dir}")


def default_sources(source_root: Path | None) -> dict[str, tuple[Path, Path]]:
    root = source_root
    if root is None:
        host_root = Path("/home/nop/Projects/nornik")
        root = host_root if host_root.exists() else Path("/source-nornik")

    return {
        "ordinary": (
            root / "Фото руд по сортам. ч1" / "Рядовые руды",
            root / "Фото руд по сортам. ч2" / "рядовые",
        ),
        "difficult": (
            root / "Фото руд по сортам. ч1" / "Труднообогатимые руды",
            root / "Фото руд по сортам. ч2" / "тонкие",
        ),
    }


def discover_source_images(sources: dict[str, tuple[Path, Path]]) -> list[SourceImage]:
    records: list[SourceImage] = []
    for label, directories in sources.items():
        for directory in directories:
            if not directory.exists():
                raise FileNotFoundError(f"Source directory does not exist: {directory}")
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(
                        SourceImage(
                            label=label,
                            label_id=LABEL_TO_ID[label],
                            source_path=path,
                            sha256=sha256_file(path),
                        )
                    )
    return records


def deduplicate(records: list[SourceImage]) -> tuple[list[SourceImage], list[list[SourceImage]], dict[str, int]]:
    by_hash: dict[str, list[SourceImage]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)

    usable: list[SourceImage] = []
    conflicts: list[list[SourceImage]] = []
    duplicate_counts = {"same_label_duplicates": 0, "conflicting_files": 0}

    for group in by_hash.values():
        labels = {record.label for record in group}
        if len(labels) > 1:
            conflicts.append(group)
            duplicate_counts["conflicting_files"] += len(group)
            continue
        duplicate_counts["same_label_duplicates"] += max(0, len(group) - 1)
        usable.append(sorted(group, key=lambda record: str(record.source_path))[0])

    return sorted(usable, key=lambda record: (record.label_id, record.sha256)), conflicts, duplicate_counts


def split_records(
    records: list[SourceImage],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[SourceImage], list[SourceImage], list[SourceImage]]:
    labels = [record.label_id for record in records]
    train_records, holdout_records = train_test_split(
        records,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    holdout_labels = [record.label_id for record in holdout_records]
    relative_val_ratio = val_ratio / (1.0 - train_ratio)
    val_records, test_records = train_test_split(
        holdout_records,
        train_size=relative_val_ratio,
        random_state=seed + 1,
        stratify=holdout_labels,
    )
    return sorted(train_records, key=sort_key), sorted(val_records, key=sort_key), sorted(test_records, key=sort_key)


def write_dataset(
    output_dir: Path,
    train_records: list[SourceImage],
    val_records: list[SourceImage],
    test_records: list[SourceImage],
) -> None:
    manifest_rows: list[dict[str, object]] = []
    for split_name, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        for record in records:
            target_dir = output_dir / split_name / record.label
            target_dir.mkdir(parents=True, exist_ok=True)
            target_name = f"{record.sha256[:16]}_{sanitize_filename(record.source_path.name)}"
            target_path = target_dir / target_name
            shutil.copy2(record.source_path, target_path)
            manifest_rows.append(
                {
                    "split": split_name,
                    "label": record.label,
                    "label_id": record.label_id,
                    "sha256": record.sha256,
                    "source_path": str(record.source_path),
                    "dataset_path": str(target_path),
                }
            )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "label", "label_id", "sha256", "source_path", "dataset_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_conflicts(path: Path, conflicts: list[list[SourceImage]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sha256", "label", "label_id", "source_path"])
        writer.writeheader()
        for group in sorted(conflicts, key=lambda items: items[0].sha256):
            for record in sorted(group, key=sort_key):
                writer.writerow(
                    {
                        "sha256": record.sha256,
                        "label": record.label,
                        "label_id": record.label_id,
                        "source_path": str(record.source_path),
                    }
                )


def write_summary(
    path: Path,
    *,
    source_records: list[SourceImage],
    unique_records: list[SourceImage],
    conflicts: list[list[SourceImage]],
    duplicate_counts: dict[str, int],
    split_records: dict[str, list[SourceImage]],
    args: argparse.Namespace,
    sources: dict[str, tuple[Path, Path]],
) -> None:
    summary = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "label_to_id": LABEL_TO_ID,
        "source_counts": count_labels(source_records),
        "usable_counts": count_labels(unique_records),
        "split_counts": {
            split_name: count_labels(records) for split_name, records in split_records.items()
        },
        "source_total": len(source_records),
        "usable_total": len(unique_records),
        "conflicting_hash_groups": len(conflicts),
        **duplicate_counts,
        "sources": {
            label: [str(path) for path in paths] for label, paths in sources.items()
        },
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def count_labels(records: list[SourceImage]) -> dict[str, int]:
    counts = Counter(record.label for record in records)
    return {label: counts.get(label, 0) for label in LABEL_TO_ID}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-zА-Яа-я0-9._-]+", "_", name).strip("._")
    return stem or "image"


def sort_key(record: SourceImage) -> tuple[int, str, str]:
    return record.label_id, record.sha256, str(record.source_path)


if __name__ == "__main__":
    main()
