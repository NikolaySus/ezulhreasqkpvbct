from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DATA_ROOT, IMAGE_EXTENSIONS, SKETCH_ROOT


@dataclass(frozen=True)
class ImagePair:
    name: str
    source_path: Path
    sketch_path: Path


def list_image_pairs() -> list[ImagePair]:
    if not DATA_ROOT.exists():
        return []

    pairs: list[ImagePair] = []
    for source_path in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name.casefold()):
        if not source_path.is_file() or source_path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        sketch_path = SKETCH_ROOT / source_path.name
        if sketch_path.is_file():
            pairs.append(ImagePair(source_path.name, source_path, sketch_path))
    return pairs


def get_image_pair(name: str) -> ImagePair:
    for pair in list_image_pairs():
        if pair.name == name:
            return pair
    raise FileNotFoundError(name)
