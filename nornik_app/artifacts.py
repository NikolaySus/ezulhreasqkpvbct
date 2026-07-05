from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import NORNIK_ROOT
from .segmentation import Approach2Settings, ClassName


CACHE_ROOT = NORNIK_ROOT / ".nornik_cache" / "color_artifacts"
CLASS_SCHEMA_VERSION = 2
CLASS_TO_CODE: dict[ClassName, int] = {
    "ore": 1,
    "matrix": 2,
    "talc": 3,
    "damage": 4,
}
CODE_TO_CLASS = {value: key for key, value in CLASS_TO_CODE.items()}


@dataclass(frozen=True)
class ColorArtifact:
    key: str
    rgb: np.ndarray
    class_map: np.ndarray
    settings: Approach2Settings


def color_artifact_key(image_name: str, source_path: Path, settings: Approach2Settings) -> str:
    stat = source_path.stat()
    payload = {
        "image": image_name,
        "source": str(source_path.resolve()),
        "sourceSize": stat.st_size,
        "sourceMtimeNs": stat.st_mtime_ns,
        "classSchema": CLASS_SCHEMA_VERSION,
        "settings": asdict(settings),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def save_color_artifact(
    key: str,
    rgb: np.ndarray,
    class_map: np.ndarray,
    settings: Approach2Settings,
) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(key)
    class_codes = class_map_to_codes(class_map)
    np.savez_compressed(
        path,
        rgb=rgb.astype(np.uint8, copy=False),
        class_codes=class_codes,
        settings_json=json.dumps(asdict(settings), sort_keys=True),
    )


def class_map_to_codes(class_map: np.ndarray) -> np.ndarray:
    return np.vectorize(CLASS_TO_CODE.__getitem__, otypes=[np.uint8])(class_map)


def codes_to_class_map(class_codes: np.ndarray) -> np.ndarray:
    class_map = np.full(class_codes.shape, "matrix", dtype=object)
    for code, class_name in CODE_TO_CLASS.items():
        class_map[class_codes == code] = class_name
    class_map[class_codes == 5] = "damage"
    return class_map


def load_color_artifact(key: str, expected_settings: Approach2Settings | None = None) -> ColorArtifact | None:
    if not _valid_key(key):
        return None
    path = _artifact_path(key)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        settings_data: dict[str, Any] = json.loads(str(data["settings_json"]))
        settings = Approach2Settings(**settings_data)
        if expected_settings is not None and settings != expected_settings:
            return None
        rgb = np.array(data["rgb"], dtype=np.uint8, copy=True)
        class_codes = np.array(data["class_codes"], dtype=np.uint8, copy=True)

    class_map = codes_to_class_map(class_codes)
    return ColorArtifact(key=key, rgb=rgb, class_map=class_map, settings=settings)


def color_artifact_available(key: str, expected_settings: Approach2Settings) -> bool:
    if not _valid_key(key):
        return False
    path = _artifact_path(key)
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        settings_data: dict[str, Any] = json.loads(str(data["settings_json"]))
    return Approach2Settings(**settings_data) == expected_settings


def _artifact_path(key: str) -> Path:
    return CACHE_ROOT / f"{key}.npz"


def _valid_key(key: str) -> bool:
    return len(key) == 24 and all(char in "0123456789abcdef" for char in key)
