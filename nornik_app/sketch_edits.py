from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from .config import SKETCH_EDITS_ROOT


def edit_file_path(image_name: str, root: Path = SKETCH_EDITS_ROOT) -> Path:
    digest = hashlib.sha1(image_name.encode("utf-8")).hexdigest()[:12]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_name).strip("_") or "image"
    return root / f"{digest}_{safe_name}.json"


def load_sketch_edit(image_name: str, root: Path = SKETCH_EDITS_ROOT) -> dict[str, Any]:
    path = edit_file_path(image_name, root)
    if not path.exists():
        return {"image": image_name, "version": 2, "segments": [], "regionMode": "inside"}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    region_mode = data.get("regionMode")
    if region_mode not in {"inside", "outside"}:
        region_mode = "inside"
    return {"image": image_name, "version": 2, "segments": segments, "regionMode": region_mode}


def save_sketch_edit(image_name: str, edit: dict[str, Any], root: Path = SKETCH_EDITS_ROOT) -> dict[str, Any]:
    normalized = {
        "image": image_name,
        "version": 2,
        "segments": list(edit.get("segments", [])),
        "regionMode": edit.get("regionMode") if edit.get("regionMode") in {"inside", "outside"} else "inside",
    }
    path = edit_file_path(image_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return normalized


def add_segment_to_edit(
    image_name: str,
    a_endpoint_id: str,
    b_endpoint_id: str,
    a: tuple[float, float],
    b: tuple[float, float],
    control: tuple[float, float],
    tangent: tuple[float, float],
    root: Path = SKETCH_EDITS_ROOT,
) -> dict[str, Any]:
    edit = load_sketch_edit(image_name, root)
    segment = {
        "id": uuid.uuid4().hex[:12],
        "aEndpointId": str(a_endpoint_id),
        "bEndpointId": str(b_endpoint_id),
        "a": [round(float(a[0]), 2), round(float(a[1]), 2)],
        "b": [round(float(b[0]), 2), round(float(b[1]), 2)],
        "routeType": "curve",
        "control": [round(float(control[0]), 2), round(float(control[1]), 2)],
        "tangent": [round(float(tangent[0]), 2), round(float(tangent[1]), 2)],
    }
    edit["segments"].append(segment)
    return save_sketch_edit(image_name, edit, root)


def delete_segment_from_edit(image_name: str, segment_id: str, root: Path = SKETCH_EDITS_ROOT) -> dict[str, Any]:
    edit = load_sketch_edit(image_name, root)
    edit["segments"] = [segment for segment in edit["segments"] if segment.get("id") != segment_id]
    return save_sketch_edit(image_name, edit, root)


def set_region_mode(image_name: str, region_mode: str, root: Path = SKETCH_EDITS_ROOT) -> dict[str, Any]:
    if region_mode not in {"inside", "outside"}:
        raise ValueError(f"Unknown region mode: {region_mode}")
    edit = load_sketch_edit(image_name, root)
    edit["regionMode"] = region_mode
    return save_sketch_edit(image_name, edit, root)
