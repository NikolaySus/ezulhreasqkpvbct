from pathlib import Path
from time import monotonic

import numpy as np
from rq import get_current_job
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image

from app.settings import SEGMENTATION_MODEL_PATH

Image.MAX_IMAGE_PIXELS = None

NUM_CLASSES = 4
CLASS_NAMES = ("ore", "matrix", "talc", "damage")
TALC_CLASS_INDEX = 2

PALETTE = np.array(
    [
        [237, 28, 36],
        [255, 242, 0],
        [63, 72, 204],
        [136, 0, 21],
    ],
    dtype=np.uint8,
)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL = None
_DEVICE = None
PROGRESS_SAVE_INTERVAL_SECONDS = 0.5


def segment_image(upload_path: str, result_path: str) -> dict[str, object]:
    progress = _ProgressReporter()
    progress.update(stage="loading_image", percent=0.0, force=True)

    source = Image.open(upload_path).convert("RGB")
    image = np.asarray(source)

    progress.update(stage="loading_model", percent=0.0, force=True)
    mask = _predict_mask(image, progress)

    progress.update(stage="saving", percent=99.0, force=True)
    result = Path(result_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_colorize_mask(mask)).save(result, format="PNG")

    talc_ratio = float((mask == TALC_CLASS_INDEX).mean())
    output = {
        "result_path": str(result),
        "classes": CLASS_NAMES,
        "talc_ratio": talc_ratio,
        "is_talcose": talc_ratio > 0.10,
    }
    progress.update(stage="finished", processed_tiles=progress.processed_tiles, percent=100.0, force=True)
    return output


def _predict_mask(image: np.ndarray, progress: "_ProgressReporter") -> np.ndarray:
    model, device = _load_model()

    h, w = image.shape[:2]
    tile_size = 256
    stride = tile_size // 2
    pad = tile_size // 2

    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]

    ys = list(range(0, ph - tile_size + 1, stride))
    xs = list(range(0, pw - tile_size + 1, stride))
    if ys[-1] != ph - tile_size:
        ys.append(ph - tile_size)
    if xs[-1] != pw - tile_size:
        xs.append(pw - tile_size)

    coords = [(y, x) for y in ys for x in xs]
    batch_size = 32 if device.type == "cuda" else 4
    window = _gaussian_window(tile_size)
    accum = np.zeros((ph, pw, NUM_CLASSES), dtype=np.float32)
    weight_sum = np.zeros((ph, pw), dtype=np.float32)
    total_tiles = len(coords)

    progress.update(
        stage="segmenting",
        processed_tiles=0,
        total_tiles=total_tiles,
        percent=0.0,
        force=True,
    )

    for offset in range(0, len(coords), batch_size):
        chunk = coords[offset : offset + batch_size]
        batch = np.stack([padded[y : y + tile_size, x : x + tile_size] for y, x in chunk], axis=0)
        probabilities = _run_batch(model, device, batch)

        for (y, x), probs in zip(chunk, probabilities):
            accum[y : y + tile_size, x : x + tile_size] += probs * window[..., None]
            weight_sum[y : y + tile_size, x : x + tile_size] += window

        processed_tiles = min(offset + len(chunk), total_tiles)
        progress.update(
            stage="segmenting",
            processed_tiles=processed_tiles,
            total_tiles=total_tiles,
            percent=(processed_tiles / total_tiles) * 100,
        )

    progress.update(stage="assembling", processed_tiles=total_tiles, total_tiles=total_tiles, percent=98.0, force=True)
    accum /= np.clip(weight_sum, 1e-6, None)[..., None]
    cropped = accum[pad : pad + h, pad : pad + w]
    return np.argmax(cropped, axis=-1).astype(np.uint8)


def _load_model():
    global _MODEL, _DEVICE

    if _MODEL is None:
        if not SEGMENTATION_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Segmentation checkpoint not found: {SEGMENTATION_MODEL_PATH}. "
                "Place ml-days-2 save_1.pth under model-artifacts/ml-days-2/saves/."
            )

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location=_DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(_DEVICE).eval()
        if _DEVICE.type == "cuda":
            model.half()
        _MODEL = model

    return _MODEL, _DEVICE


@torch.inference_mode()
def _run_batch(model, device: torch.device, batch: np.ndarray) -> np.ndarray:
    tensor = _preprocess_batch(batch).to(device)
    if device.type == "cuda":
        tensor = tensor.half()
    logits = model(tensor)
    probabilities = F.softmax(logits.float(), dim=1)
    return probabilities.permute(0, 2, 3, 1).cpu().numpy()


def _preprocess_batch(batch: np.ndarray) -> torch.Tensor:
    tensor = batch.astype(np.float32) / 255.0
    tensor = (tensor - MEAN) / STD
    return torch.from_numpy(tensor).permute(0, 3, 1, 2).contiguous()


def _gaussian_window(tile_size: int, sigma_frac: float = 0.125) -> np.ndarray:
    axis = np.arange(tile_size) - (tile_size - 1) / 2.0
    one_dimensional = np.exp(-(axis**2) / (2 * (tile_size * sigma_frac) ** 2))
    window = np.outer(one_dimensional, one_dimensional).astype(np.float32)
    return np.clip(window, 1e-3, None)


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1).astype(np.int64)]


class _ProgressReporter:
    def __init__(self) -> None:
        self.job = get_current_job()
        self.last_saved_at = 0.0
        self.processed_tiles = 0
        self.total_tiles = None

    def update(
        self,
        *,
        stage: str,
        processed_tiles: int | None = None,
        total_tiles: int | None = None,
        percent: float | None = None,
        force: bool = False,
    ) -> None:
        if self.job is None:
            return

        if processed_tiles is not None:
            self.processed_tiles = processed_tiles
        if total_tiles is not None:
            self.total_tiles = total_tiles

        now = monotonic()
        if not force and now - self.last_saved_at < PROGRESS_SAVE_INTERVAL_SECONDS:
            return

        progress: dict[str, object] = {"stage": stage}
        if self.total_tiles is not None:
            progress["total_tiles"] = self.total_tiles
            progress["processed_tiles"] = self.processed_tiles
        if percent is not None:
            progress["percent"] = max(0.0, min(100.0, round(float(percent), 1)))

        self.job.meta["progress"] = progress
        self.job.save_meta()
        self.last_saved_at = now
