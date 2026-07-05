from __future__ import annotations

import json
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from .artifacts import CLASS_TO_CODE, class_map_to_codes, codes_to_class_map
from .config import EXPORTS_ROOT
from .data import list_image_pairs
from .image_settings import load_image_settings
from .segmentation import (
    CORRECTED_OVERLAY_COLORS,
    _approach2_class_map_with_illumination_status,
    _overlay_array,
    corrected_approach2_data_from_class_map,
    parse_approach2,
    parse_correction_settings,
    parse_sketch_settings,
)
from .sketch_edits import load_sketch_edit


ExportKind = Literal[
    "color-masks",
    "color-previews",
    "corrected-masks",
    "corrected-previews",
    "corrected-originals",
    "all",
]


class ExportJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._sequence = 0

    def start(self, kind: ExportKind) -> dict[str, Any]:
        job_id = self._next_id(kind)
        job = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "processed": 0,
            "total": len(list_image_pairs()),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "downloadUrl": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job_id, kind), daemon=True)
        thread.start()
        return dict(job)

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def path(self, job_id: str) -> Path | None:
        job = self.status(job_id)
        if not job or job.get("status") != "done":
            return None
        path = EXPORTS_ROOT / f"{job_id}.zip"
        return path if path.is_file() else None

    def _next_id(self, kind: str) -> str:
        with self._lock:
            self._sequence += 1
            return f"{kind}-{int(time.time())}-{self._sequence}"

    def _run(self, job_id: str, kind: ExportKind) -> None:
        try:
            path, manifest = build_export_archive(kind, job_id, self._progress_callback(job_id))
            with self._lock:
                self._jobs[job_id].update(
                    {
                        "status": "done",
                        "processed": manifest["processedCount"],
                        "success": manifest["successCount"],
                        "skipped": manifest["skippedCount"],
                        "failed": manifest["failureCount"],
                        "downloadUrl": f"/api/export-jobs/{job_id}/download",
                        "path": str(path),
                    }
                )
        except Exception:
            with self._lock:
                self._jobs[job_id].update({"status": "error", "error": traceback.format_exc()})

    def _progress_callback(self, job_id: str):
        def update(processed: int, total: int, image_name: str) -> None:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].update({"processed": processed, "total": total, "currentImage": image_name})

        return update


def build_export_archive(kind: ExportKind, job_id: str, progress_callback=None) -> tuple[Path, dict[str, Any]]:
    if kind == "all":
        return build_all_export_archive(job_id, progress_callback)

    EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = EXPORTS_ROOT / f"{job_id}.zip"
    pairs = list_image_pairs()
    records: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, pair in enumerate(pairs, start=1):
            if progress_callback:
                progress_callback(index - 1, len(pairs), pair.name)
            record_started_at = time.perf_counter()
            record: dict[str, Any] = {"image": pair.name}
            try:
                image_settings = load_image_settings(pair.name)
                if image_settings.get("excluded_from_exports"):
                    record.update({"status": "skipped", "reason": "excluded", "settings": image_settings})
                    records.append(_finalized_record(record, record_started_at))
                    if progress_callback:
                        progress_callback(index, len(pairs), pair.name)
                    continue
                with Image.open(pair.source_path) as source:
                    original_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
                    source_rgb, class_map, illumination_applied = _approach2_class_map_with_illumination_status(
                        source,
                        parse_approach2(_string_params(image_settings["a2"])),
                    )
                    if kind.startswith("color-"):
                        _write_color_item(archive, kind, pair.name, original_rgb, class_map)
                        record.update(
                            {
                                "status": "ok",
                                "shape": [int(original_rgb.shape[0]), int(original_rgb.shape[1])],
                                "illuminationCorrectionApplied": bool(illumination_applied),
                            }
                        )
                    else:
                        sketch_edit = load_sketch_edit(pair.name)
                        with Image.open(pair.sketch_path) as sketch:
                            data = corrected_approach2_data_from_class_map(
                                source,
                                sketch,
                                source_rgb,
                                class_map,
                                parse_approach2(_string_params(image_settings["a2"])),
                                parse_sketch_settings(_string_params(image_settings["sketch"])),
                                sketch_edit["segments"],
                                sketch_edit["regionMode"],
                                parse_correction_settings(_string_params(image_settings["correction"])),
                            )
                        if not data["correction_applied"]:
                            record.update({"status": "skipped", "reason": "open-sketch"})
                        else:
                            crop_box = _scaled_crop_box(data["crop_box"], data["class_map"].shape, original_rgb.shape)
                            _write_corrected_item(archive, kind, pair.name, original_rgb, data, crop_box)
                            left, top, right, bottom = crop_box
                            record.update(
                                {
                                    "status": "ok",
                                    "shape": [int(bottom - top), int(right - left)],
                                    "cropBox": [int(left), int(top), int(right), int(bottom)],
                                    "correctionDetails": data["details"],
                                }
                            )
                record["settings"] = image_settings
            except Exception as exc:
                record.update({"status": "error", "error": str(exc)})
            records.append(_finalized_record(record, record_started_at))
            if progress_callback:
                progress_callback(index, len(pairs), pair.name)

        manifest = {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "totalImages": len(records),
            "processedCount": len(records),
            "successCount": sum(1 for record in records if record["status"] == "ok"),
            "skippedCount": sum(1 for record in records if record["status"] == "skipped"),
            "failureCount": sum(1 for record in records if record["status"] == "error"),
            "elapsedSeconds": round(time.perf_counter() - started_at, 3),
            "classMapping": dict(CLASS_TO_CODE),
            "images": records,
        }
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return output_path, manifest


def build_all_export_archive(job_id: str, progress_callback=None) -> tuple[Path, dict[str, Any]]:
    EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = EXPORTS_ROOT / f"{job_id}.zip"
    pairs = list_image_pairs()
    records: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, pair in enumerate(pairs, start=1):
            if progress_callback:
                progress_callback(index - 1, len(pairs), pair.name)
            record_started_at = time.perf_counter()
            record: dict[str, Any] = {"image": pair.name}
            try:
                image_settings = load_image_settings(pair.name)
                record["settings"] = image_settings
                if image_settings.get("excluded_from_exports"):
                    record.update(
                        {
                            "status": "skipped",
                            "reason": "excluded",
                            "color": {"status": "skipped", "reason": "excluded"},
                            "corrected": {"status": "skipped", "reason": "excluded"},
                        }
                    )
                    records.append(_finalized_record(record, record_started_at))
                    if progress_callback:
                        progress_callback(index, len(pairs), pair.name)
                    continue

                approach_settings = parse_approach2(_string_params(image_settings["a2"]))
                with Image.open(pair.source_path) as source:
                    original_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
                    source_rgb, class_map, illumination_applied = _approach2_class_map_with_illumination_status(
                        source,
                        approach_settings,
                    )
                    color_record = _write_all_color_items(archive, pair.name, original_rgb, class_map)
                    color_record["illuminationCorrectionApplied"] = bool(illumination_applied)

                    sketch_edit = load_sketch_edit(pair.name)
                    with Image.open(pair.sketch_path) as sketch:
                        data = corrected_approach2_data_from_class_map(
                            source,
                            sketch,
                            source_rgb,
                            class_map,
                            approach_settings,
                            parse_sketch_settings(_string_params(image_settings["sketch"])),
                            sketch_edit["segments"],
                            sketch_edit["regionMode"],
                            parse_correction_settings(_string_params(image_settings["correction"])),
                        )

                    if not data["correction_applied"]:
                        corrected_record = {"status": "skipped", "reason": "open-sketch"}
                    else:
                        crop_box = _scaled_crop_box(data["crop_box"], data["class_map"].shape, original_rgb.shape)
                        corrected_record = _write_all_corrected_items(archive, pair.name, original_rgb, data, crop_box)
                        corrected_record["correctionDetails"] = data["details"]

                corrected_status = str(corrected_record["status"])
                record.update(
                    {
                        "status": "ok" if corrected_status in {"ok", "skipped"} else corrected_status,
                        "color": color_record,
                        "corrected": corrected_record,
                    }
                )
            except Exception as exc:
                record.update({"status": "error", "error": str(exc)})
            records.append(_finalized_record(record, record_started_at))
            if progress_callback:
                progress_callback(index, len(pairs), pair.name)

        manifest = {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "kind": "all",
            "totalImages": len(records),
            "processedCount": len(records),
            "successCount": sum(1 for record in records if record["status"] == "ok"),
            "skippedCount": sum(1 for record in records if record["status"] == "skipped"),
            "failureCount": sum(1 for record in records if record["status"] == "error"),
            "elapsedSeconds": round(time.perf_counter() - started_at, 3),
            "classMapping": dict(CLASS_TO_CODE),
            "images": records,
        }
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return output_path, manifest


def _finalized_record(record: dict[str, Any], started_at: float) -> dict[str, Any]:
    record["elapsedSeconds"] = round(time.perf_counter() - started_at, 3)
    return record


def _write_color_item(
    archive: zipfile.ZipFile,
    kind: ExportKind,
    image_name: str,
    rgb: np.ndarray,
    class_map: np.ndarray,
) -> None:
    stem = Path(image_name).stem
    full_size_map = _class_map_at_rgb_size(class_map, rgb)
    if kind == "color-masks":
        archive.writestr(f"masks/{stem}.npy", _npy_bytes(class_map_to_codes(full_size_map)))
    elif kind == "color-previews":
        archive.writestr(f"previews/{stem}.png", _png_bytes(_overlay_array(rgb, full_size_map)))
    else:
        raise ValueError(f"Unsupported color export kind: {kind}")


def _write_corrected_item(
    archive: zipfile.ZipFile,
    kind: ExportKind,
    image_name: str,
    rgb: np.ndarray,
    data: dict[str, Any],
    crop_box: tuple[int, int, int, int],
) -> None:
    stem = Path(image_name).stem
    left, top, right, bottom = crop_box
    full_size_map = _class_map_at_rgb_size(data["class_map"], rgb)
    cropped_map = full_size_map[top:bottom, left:right]
    cropped_rgb = rgb[top:bottom, left:right]
    if kind == "corrected-masks":
        archive.writestr(f"masks/{stem}.npy", _npy_bytes(class_map_to_codes(cropped_map)))
    elif kind == "corrected-previews":
        overlay = _overlay_array(cropped_rgb, cropped_map, CORRECTED_OVERLAY_COLORS)
        archive.writestr(f"previews/{stem}.png", _png_bytes(overlay))
    elif kind == "corrected-originals":
        archive.writestr(f"originals/{stem}.png", _png_bytes(cropped_rgb))
    else:
        raise ValueError(f"Unsupported corrected export kind: {kind}")


def _write_all_color_items(
    archive: zipfile.ZipFile,
    image_name: str,
    rgb: np.ndarray,
    class_map: np.ndarray,
) -> dict[str, Any]:
    stem = Path(image_name).stem
    full_size_map = _class_map_at_rgb_size(class_map, rgb)
    archive.writestr(f"color-masks/{stem}.npy", _npy_bytes(class_map_to_codes(full_size_map)))
    archive.writestr(f"color-previews/{stem}.png", _png_bytes(_overlay_array(rgb, full_size_map)))
    return {
        "status": "ok",
        "shape": [int(rgb.shape[0]), int(rgb.shape[1])],
        "maskFile": f"color-masks/{stem}.npy",
        "previewFile": f"color-previews/{stem}.png",
    }


def _write_all_corrected_items(
    archive: zipfile.ZipFile,
    image_name: str,
    rgb: np.ndarray,
    data: dict[str, Any],
    crop_box: tuple[int, int, int, int],
) -> dict[str, Any]:
    stem = Path(image_name).stem
    left, top, right, bottom = crop_box
    full_size_map = _class_map_at_rgb_size(data["class_map"], rgb)
    cropped_map = full_size_map[top:bottom, left:right]
    cropped_rgb = rgb[top:bottom, left:right]
    archive.writestr(f"corrected-masks/{stem}.npy", _npy_bytes(class_map_to_codes(cropped_map)))
    archive.writestr(f"corrected-previews/{stem}.png", _png_bytes(_overlay_array(cropped_rgb, cropped_map, CORRECTED_OVERLAY_COLORS)))
    archive.writestr(f"corrected-originals/{stem}.png", _png_bytes(cropped_rgb))
    return {
        "status": "ok",
        "shape": [int(bottom - top), int(right - left)],
        "cropBox": [int(left), int(top), int(right), int(bottom)],
        "maskFile": f"corrected-masks/{stem}.npy",
        "previewFile": f"corrected-previews/{stem}.png",
        "originalFile": f"corrected-originals/{stem}.png",
    }


def _class_map_at_rgb_size(class_map: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if class_map.shape == (height, width):
        return class_map
    codes = class_map_to_codes(class_map)
    resized = Image.fromarray(codes).resize((width, height), Image.Resampling.NEAREST)
    return codes_to_class_map(np.asarray(resized, dtype=np.uint8))


def _scaled_crop_box(
    box: tuple[int, int, int, int],
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    source_height, source_width = int(source_shape[0]), int(source_shape[1])
    target_height, target_width = int(target_shape[0]), int(target_shape[1])
    left, top, right, bottom = box
    x_scale = target_width / float(source_width)
    y_scale = target_height / float(source_height)
    scaled_left = max(0, min(target_width - 1, int(np.floor(left * x_scale))))
    scaled_top = max(0, min(target_height - 1, int(np.floor(top * y_scale))))
    scaled_right = max(scaled_left + 1, min(target_width, int(np.ceil(right * x_scale))))
    scaled_bottom = max(scaled_top + 1, min(target_height, int(np.ceil(bottom * y_scale))))
    return scaled_left, scaled_top, scaled_right, scaled_bottom


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array.astype(np.uint8, copy=False), allow_pickle=False)
    return buffer.getvalue()


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _string_params(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}
