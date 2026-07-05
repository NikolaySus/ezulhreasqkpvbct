from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifacts import CLASS_TO_CODE, codes_to_class_map
from .batch_color_masks import DEFAULT_OUTPUT_ROOT as DEFAULT_MASK_ROOT
from .batch_color_masks import list_source_images
from .config import COLOR_MARKUP_PREVIEWS_ROOT
from .segmentation import Approach2Settings, _overlay_array, _prepare_rgb, _stats, approach2_defaults


DEFAULT_OUTPUT_ROOT = COLOR_MARKUP_PREVIEWS_ROOT / "approach2_default"


def run_batch(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    mask_root: Path = DEFAULT_MASK_ROOT,
    source_paths: list[Path] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    mask_manifest = _load_mask_manifest(mask_root)
    settings = _settings_from_manifest(mask_manifest)
    sources = _source_map(source_paths if source_paths is not None else list_source_images())
    mask_records = _mask_records(mask_root, mask_manifest)
    started_at = time.perf_counter()
    records: list[dict[str, Any]] = []

    for mask_record in mask_records:
        record_started_at = time.perf_counter()
        image_name = str(mask_record["image"])
        record: dict[str, Any] = {"image": image_name, "maskFile": str(mask_record["maskFile"])}
        try:
            source_path = sources[image_name]
            mask_path = mask_root / str(mask_record["maskFile"])
            mask = np.load(mask_path, allow_pickle=False)
            if mask.dtype != np.uint8:
                raise ValueError(f"Mask must be uint8, got {mask.dtype}")
            if mask.ndim != 2:
                raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

            with Image.open(source_path) as image:
                rgb = _prepare_rgb(image, settings.max_work_side)
                if rgb.shape[:2] != mask.shape:
                    rgb = _resize_rgb_to_mask(image, mask)

            class_map = codes_to_class_map(mask)
            overlay = _overlay_array(rgb, class_map)
            preview_path = output_root / f"{Path(image_name).stem}.png"
            Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(preview_path, format="PNG", optimize=True)
            record.update(
                {
                    "status": "ok",
                    "previewFile": preview_path.name,
                    "shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "sourceShape": [int(rgb.shape[0]), int(rgb.shape[1])],
                    "stats": _stats(class_map),
                    "illuminationCorrectionApplied": mask_record.get("illuminationCorrectionApplied"),
                    "elapsedSeconds": round(time.perf_counter() - record_started_at, 3),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "elapsedSeconds": round(time.perf_counter() - record_started_at, 3),
                }
            )
        records.append(record)

    success_count = sum(1 for record in records if record["status"] == "ok")
    manifest = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "outputRoot": str(output_root),
        "maskRoot": str(mask_root),
        "totalImages": len(records),
        "successCount": success_count,
        "failureCount": len(records) - success_count,
        "elapsedSeconds": round(time.perf_counter() - started_at, 3),
        "previewFormat": "png-overlay-from-mask",
        "sourceMaskFormat": mask_manifest.get("maskFormat"),
        "classMapping": CLASS_TO_CODE,
        "settings": asdict(settings),
        "images": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _load_mask_manifest(mask_root: Path) -> dict[str, Any]:
    manifest_path = mask_root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _settings_from_manifest(mask_manifest: dict[str, Any]) -> Approach2Settings:
    settings_data = mask_manifest.get("settings")
    if not isinstance(settings_data, dict):
        return approach2_defaults()
    return Approach2Settings(**settings_data)


def _source_map(source_paths: list[Path]) -> dict[str, Path]:
    return {path.name: path for path in source_paths}


def _mask_records(mask_root: Path, mask_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_images = mask_manifest.get("images")
    if isinstance(manifest_images, list):
        records = [
            record
            for record in manifest_images
            if isinstance(record, dict) and record.get("status") == "ok" and record.get("maskFile")
        ]
        if records:
            return records
    return [{"image": f"{path.stem}.JPG", "maskFile": path.name} for path in sorted(mask_root.glob("*.npy"))]


def _resize_rgb_to_mask(image: Image.Image, mask: np.ndarray) -> np.ndarray:
    resized = image.convert("RGB").resize((int(mask.shape[1]), int(mask.shape[0])), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def main() -> int:
    manifest = run_batch()
    print(
        json.dumps(
            {
                "outputRoot": manifest["outputRoot"],
                "maskRoot": manifest["maskRoot"],
                "totalImages": manifest["totalImages"],
                "successCount": manifest["successCount"],
                "failureCount": manifest["failureCount"],
                "elapsedSeconds": manifest["elapsedSeconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if manifest["failureCount"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
