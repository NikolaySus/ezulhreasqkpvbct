from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .artifacts import CLASS_TO_CODE, class_map_to_codes, codes_to_class_map
from .config import COLOR_MARKUP_MASKS_ROOT, DATA_ROOT, IMAGE_EXTENSIONS
from .segmentation import _approach2_class_map_with_illumination_status, approach2_defaults, kmeans_backend_name


DEFAULT_OUTPUT_ROOT = COLOR_MARKUP_MASKS_ROOT / "approach2_default"


def list_source_images(data_root: Path = DATA_ROOT) -> list[Path]:
    if not data_root.exists():
        return []
    return [
        path
        for path in sorted(data_root.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]


def run_batch(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_paths: list[Path] | None = None,
) -> dict[str, Any]:
    settings = approach2_defaults()
    output_root.mkdir(parents=True, exist_ok=True)
    sources = source_paths if source_paths is not None else list_source_images()
    started_at = time.perf_counter()
    records: list[dict[str, Any]] = []

    for source_path in sources:
        record_started_at = time.perf_counter()
        record: dict[str, Any] = {"image": source_path.name}
        try:
            with Image.open(source_path) as image:
                original_size = image.size
                rgb, class_map, illumination_applied = _approach2_class_map_with_illumination_status(image, settings)
            class_map = _class_map_at_size(class_map, original_size)
            mask = class_map_to_codes(class_map)
            mask_path = output_root / f"{source_path.stem}.npy"
            np.save(mask_path, mask, allow_pickle=False)
            record.update(
                {
                    "status": "ok",
                    "maskFile": mask_path.name,
                    "shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "sourceShape": [int(original_size[1]), int(original_size[0])],
                    "illuminationCorrectionApplied": bool(illumination_applied),
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
        "totalImages": len(records),
        "successCount": success_count,
        "failureCount": len(records) - success_count,
        "elapsedSeconds": round(time.perf_counter() - started_at, 3),
        "maskFormat": "uint8-npy-uncompressed",
        "classMapping": dict(CLASS_TO_CODE),
        "settings": asdict(settings),
        "backend": {
            "kmeans": kmeans_backend_name(),
            "opencvCudaDeviceCount": _opencv_cuda_device_count(),
        },
        "images": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _opencv_cuda_device_count() -> int:
    if not hasattr(cv2, "cuda"):
        return 0
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def _class_map_at_size(class_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    if class_map.shape == (height, width):
        return class_map
    codes = class_map_to_codes(class_map)
    resized = Image.fromarray(codes).resize((width, height), Image.Resampling.NEAREST)
    return codes_to_class_map(np.asarray(resized, dtype=np.uint8))


def main() -> int:
    manifest = run_batch()
    print(
        json.dumps(
            {
                "outputRoot": manifest["outputRoot"],
                "totalImages": manifest["totalImages"],
                "successCount": manifest["successCount"],
                "failureCount": manifest["failureCount"],
                "elapsedSeconds": manifest["elapsedSeconds"],
                "backend": manifest["backend"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if manifest["failureCount"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
