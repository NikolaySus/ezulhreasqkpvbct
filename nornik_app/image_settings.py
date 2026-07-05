from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .config import IMAGE_SETTINGS_ROOT
from .data import get_image_pair
from .segmentation import (
    approach2_defaults,
    correction_defaults,
    parse_approach2,
    parse_correction_settings,
    parse_sketch_settings,
    sketch_defaults,
)


def default_image_settings(image_name: str) -> dict[str, Any]:
    return {
        "image": image_name,
        "a2": asdict(approach2_defaults()),
        "sketch": asdict(sketch_defaults()),
        "correction": asdict(correction_defaults()),
        "excluded_from_exports": False,
        "customized": False,
    }


def load_image_settings(image_name: str) -> dict[str, Any]:
    get_image_pair(image_name)
    defaults = default_image_settings(image_name)
    path = image_settings_path(image_name)
    if not path.is_file():
        return defaults
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(raw, dict):
        return defaults
    return normalize_image_settings(image_name, raw, customized=True)


def save_image_settings(image_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    get_image_pair(image_name)
    normalized = normalize_image_settings(image_name, payload, customized=True)
    IMAGE_SETTINGS_ROOT.mkdir(parents=True, exist_ok=True)
    image_settings_path(image_name).write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return normalized


def delete_image_settings(image_name: str) -> dict[str, Any]:
    get_image_pair(image_name)
    path = image_settings_path(image_name)
    if path.exists():
        path.unlink()
    return default_image_settings(image_name)


def image_settings_path(image_name: str):
    digest = hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:12]
    return IMAGE_SETTINGS_ROOT / f"{digest}.json"


def normalize_image_settings(image_name: str, payload: dict[str, Any], customized: bool) -> dict[str, Any]:
    defaults = default_image_settings(image_name)
    a2_source = _section(payload, "a2", "approach") or defaults["a2"]
    sketch_source = _section(payload, "sketch") or defaults["sketch"]
    correction_source = _section(payload, "correction", "corrected") or defaults["correction"]
    return {
        "image": image_name,
        "a2": asdict(parse_approach2(_string_params(a2_source))),
        "sketch": asdict(parse_sketch_settings(_string_params(sketch_source))),
        "correction": asdict(parse_correction_settings(_string_params(correction_source))),
        "excluded_from_exports": _bool_value(payload.get("excluded_from_exports"), False),
        "customized": customized,
    }


def _section(payload: dict[str, Any], *names: str) -> dict[str, Any] | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, dict):
            return value
    settings = payload.get("settings")
    target = payload.get("target")
    if isinstance(settings, dict):
        if "approach" in names and target == "approach":
            return settings
        if "sketch" in names and target == "sketch":
            return settings
        if ("correction" in names or "corrected" in names) and target == "corrected":
            return settings
    return None


def _string_params(values: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in values.items()}


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default
