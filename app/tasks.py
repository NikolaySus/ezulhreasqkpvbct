from pathlib import Path
from time import monotonic

import numpy as np
from rq import get_current_job
import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from app.settings import INFERENCE_MODEL_PATH

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
_CLASSIFICATION_HEAD = None
_DEVICE = None
PROGRESS_SAVE_INTERVAL_SECONDS = 0.5
TALCOSE_THRESHOLD = 0.10
DIFFICULT_THRESHOLD = 0.50
MAX_LABELED_HEATMAP_TILES = 400


def segment_image(upload_path: str, result_path: str) -> dict[str, object]:
    progress = _ProgressReporter()
    progress.update(stage="loading_image", percent=0.0, force=True)

    source = Image.open(upload_path).convert("RGB")
    image = np.asarray(source)

    progress.update(stage="loading_model", percent=0.0, force=True)
    prediction = _predict(image, progress)

    progress.update(stage="saving", percent=99.0, force=True)
    segmentation_result = Path(result_path)
    heatmap_result = segmentation_result.with_name(f"{segmentation_result.stem}-difficulty-heatmap.png")
    segmentation_result.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_colorize_mask(prediction.mask)).save(segmentation_result, format="PNG")
    Image.fromarray(
        _render_difficulty_heatmap(
            image,
            prediction.difficulty_map,
            prediction.tile_probabilities,
            prediction.tile_centers,
        )
    ).save(heatmap_result, format="PNG")

    talc_ratio = float((prediction.mask == TALC_CLASS_INDEX).mean())
    difficulty_probability = float(np.mean(prediction.tile_probabilities)) if prediction.tile_probabilities else 0.0
    is_talcose = talc_ratio > TALCOSE_THRESHOLD
    is_difficult = difficulty_probability >= DIFFICULT_THRESHOLD
    output = {
        "result_path": str(segmentation_result),
        "segmentation_path": str(segmentation_result),
        "difficulty_heatmap_path": str(heatmap_result),
        "classes": CLASS_NAMES,
        "classification_classes": ("ordinary", "difficult"),
        "talc_ratio": talc_ratio,
        "is_talcose": is_talcose,
        "difficulty_probability": difficulty_probability,
        "is_difficult": is_difficult,
        "verdict": _verdict(is_talcose, is_difficult),
    }
    progress.update(stage="finished", processed_tiles=progress.processed_tiles, percent=100.0, force=True)
    return output


class _Prediction:
    def __init__(
        self,
        *,
        mask: np.ndarray,
        difficulty_map: np.ndarray,
        tile_probabilities: list[float],
        tile_centers: list[tuple[int, int]],
    ) -> None:
        self.mask = mask
        self.difficulty_map = difficulty_map
        self.tile_probabilities = tile_probabilities
        self.tile_centers = tile_centers


class _ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(2048, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.linear(self.dropout(pooled)).squeeze(1)


def _predict(image: np.ndarray, progress: "_ProgressReporter") -> _Prediction:
    model, classification_head, device = _load_model()

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
    difficulty_accum = np.zeros((ph, pw), dtype=np.float32)
    weight_sum = np.zeros((ph, pw), dtype=np.float32)
    total_tiles = len(coords)
    tile_probabilities: list[float] = []
    tile_centers: list[tuple[int, int]] = []

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
        probabilities, difficulty_probabilities = _run_batch(model, classification_head, device, batch)

        for (y, x), probs, difficulty_probability in zip(chunk, probabilities, difficulty_probabilities):
            accum[y : y + tile_size, x : x + tile_size] += probs * window[..., None]
            difficulty_accum[y : y + tile_size, x : x + tile_size] += float(difficulty_probability) * window
            weight_sum[y : y + tile_size, x : x + tile_size] += window
            center_x = int(round(x + tile_size / 2 - pad))
            center_y = int(round(y + tile_size / 2 - pad))
            if 0 <= center_x < w and 0 <= center_y < h:
                tile_centers.append((center_x, center_y))
                tile_probabilities.append(float(difficulty_probability))

        processed_tiles = min(offset + len(chunk), total_tiles)
        progress.update(
            stage="segmenting",
            processed_tiles=processed_tiles,
            total_tiles=total_tiles,
            percent=(processed_tiles / total_tiles) * 100,
        )

    progress.update(stage="assembling", processed_tiles=total_tiles, total_tiles=total_tiles, percent=98.0, force=True)
    accum /= np.clip(weight_sum, 1e-6, None)[..., None]
    difficulty_accum /= np.clip(weight_sum, 1e-6, None)
    cropped = accum[pad : pad + h, pad : pad + w]
    cropped_difficulty = difficulty_accum[pad : pad + h, pad : pad + w]
    return _Prediction(
        mask=np.argmax(cropped, axis=-1).astype(np.uint8),
        difficulty_map=np.clip(cropped_difficulty, 0.0, 1.0),
        tile_probabilities=tile_probabilities,
        tile_centers=tile_centers,
    )


def _load_model():
    global _MODEL, _CLASSIFICATION_HEAD, _DEVICE

    if _MODEL is None or _CLASSIFICATION_HEAD is None:
        if not INFERENCE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Inference checkpoint not found: {INFERENCE_MODEL_PATH}. "
                "Place recommended.pth under model-artifacts/ml-days-2/classification/."
            )

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        classification_head = _ClassificationHead()
        checkpoint = torch.load(INFERENCE_MODEL_PATH, map_location=_DEVICE, weights_only=False)
        if "classification_head_state_dict" not in checkpoint:
            raise KeyError("Inference checkpoint must contain classification_head_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"])
        classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
        model.to(_DEVICE).eval()
        classification_head.to(_DEVICE).eval()
        if _DEVICE.type == "cuda":
            model.half()
            classification_head.half()
        _MODEL = model
        _CLASSIFICATION_HEAD = classification_head

    return _MODEL, _CLASSIFICATION_HEAD, _DEVICE


@torch.inference_mode()
def _run_batch(
    model: smp.DeepLabV3Plus,
    classification_head: _ClassificationHead,
    device: torch.device,
    batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = _preprocess_batch(batch).to(device)
    if device.type == "cuda":
        tensor = tensor.half()
    features = model.encoder(tensor)
    decoder_output = model.decoder(features)
    segmentation_logits = model.segmentation_head(decoder_output)
    classification_logits = classification_head(features[-1])
    segmentation_probabilities = F.softmax(segmentation_logits.float(), dim=1)
    difficulty_probabilities = torch.sigmoid(classification_logits.float())
    return (
        segmentation_probabilities.permute(0, 2, 3, 1).cpu().numpy(),
        difficulty_probabilities.cpu().numpy(),
    )


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


def _render_difficulty_heatmap(
    image: np.ndarray,
    difficulty_map: np.ndarray,
    tile_probabilities: list[float],
    tile_centers: list[tuple[int, int]],
) -> np.ndarray:
    heat_colors = _difficulty_colormap(difficulty_map)
    overlay = (image.astype(np.float32) * 0.45 + heat_colors.astype(np.float32) * 0.55).astype(np.uint8)

    if len(tile_probabilities) <= MAX_LABELED_HEATMAP_TILES:
        canvas = Image.fromarray(overlay)
        draw = ImageDraw.Draw(canvas)
        for (x, y), probability in zip(tile_centers, tile_probabilities):
            label = f"{probability:.2f}"
            bbox = draw.textbbox((x, y), label, anchor="mm")
            padding = 2
            draw.rectangle(
                (
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding,
                ),
                fill=(15, 23, 42),
            )
            draw.text((x, y), label, fill=(255, 255, 255), anchor="mm")
        overlay = np.asarray(canvas)

    return overlay


def _difficulty_colormap(probabilities: np.ndarray) -> np.ndarray:
    bounded = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
    low = np.array([16, 185, 129], dtype=np.float32)
    mid = np.array([250, 204, 21], dtype=np.float32)
    high = np.array([220, 38, 38], dtype=np.float32)

    lower_half = bounded <= 0.5
    t_low = np.clip(bounded / 0.5, 0.0, 1.0)[..., None]
    t_high = np.clip((bounded - 0.5) / 0.5, 0.0, 1.0)[..., None]
    colors = np.empty((*bounded.shape, 3), dtype=np.float32)
    colors[lower_half] = (low + (mid - low) * t_low)[lower_half]
    colors[~lower_half] = (mid + (high - mid) * t_high)[~lower_half]
    return colors.astype(np.uint8)


def _verdict(is_talcose: bool, is_difficult: bool) -> str:
    if is_talcose:
        return "оталькованная"
    return "труднообогатимая" if is_difficult else "рядовая"


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
