from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from rq import get_current_job
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from app.settings import INFERENCE_MODEL_PATH
from training.train_global_context_segformer import GlobalContextSegformer, global_roi_for_patch

Image.MAX_IMAGE_PIXELS = None

NUM_CLASSES = 4
CLASS_NAMES = ("ore", "matrix", "talc", "damage")
ORE_CLASS_INDEX = 0
TALC_CLASS_INDEX = 2
TILE_SIZE = 512
TILE_STRIDE = 192
BATCH_SIZE_CUDA = 8
BATCH_SIZE_CPU = 1
GRAY_MEAN = 0.449
GRAY_STD = 0.226

PALETTE = np.array(
    [
        [237, 28, 36],
        [255, 242, 0],
        [63, 72, 204],
        [136, 0, 21],
    ],
    dtype=np.uint8,
)

_MODEL = None
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
            prediction.difficulty_coverage,
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
        difficulty_coverage: np.ndarray,
        tile_probabilities: list[float],
        tile_centers: list[tuple[int, int]],
    ) -> None:
        self.mask = mask
        self.difficulty_map = difficulty_map
        self.difficulty_coverage = difficulty_coverage
        self.tile_probabilities = tile_probabilities
        self.tile_centers = tile_centers


def _predict(image: np.ndarray, progress: "_ProgressReporter") -> _Prediction:
    model, device = _load_model()

    h, w = image.shape[:2]
    tile_size = TILE_SIZE
    stride = TILE_STRIDE
    pad = tile_size // 2

    gray = _to_normalized_gray_image(image)
    global_feature = _global_feature(model, device, gray)
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="reflect")
    ph, pw = padded.shape[:2]

    ys = list(range(0, ph - tile_size + 1, stride))
    xs = list(range(0, pw - tile_size + 1, stride))
    if ys[-1] != ph - tile_size:
        ys.append(ph - tile_size)
    if xs[-1] != pw - tile_size:
        xs.append(pw - tile_size)

    coords = [(y, x) for y in ys for x in xs]
    batch_size = BATCH_SIZE_CUDA if device.type == "cuda" else BATCH_SIZE_CPU
    window = _gaussian_window(tile_size)
    accum = np.zeros((ph, pw, NUM_CLASSES), dtype=np.float32)
    difficulty_accum = np.zeros((ph, pw), dtype=np.float32)
    difficulty_weight_sum = np.zeros((ph, pw), dtype=np.float32)
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
        rois = np.array(
            [
                global_roi_for_patch(
                    x - pad,
                    y - pad,
                    tile_size,
                    tile_size,
                    w,
                    h,
                )
                for y, x in chunk
            ],
            dtype=np.float32,
        )
        probabilities, difficulty_probabilities = _run_batch(model, device, batch, global_feature, rois)

        for (y, x), probs, difficulty_probability in zip(chunk, probabilities, difficulty_probabilities):
            accum[y : y + tile_size, x : x + tile_size] += probs * window[..., None]
            weight_sum[y : y + tile_size, x : x + tile_size] += window
            has_ore = _tile_intersects_predicted_ore(probs, y, x, pad, h, w)
            if has_ore:
                difficulty_accum[y : y + tile_size, x : x + tile_size] += float(difficulty_probability) * window
                difficulty_weight_sum[y : y + tile_size, x : x + tile_size] += window
            center_x = int(round(x + tile_size / 2 - pad))
            center_y = int(round(y + tile_size / 2 - pad))
            if has_ore and 0 <= center_x < w and 0 <= center_y < h:
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
    difficulty_accum /= np.clip(difficulty_weight_sum, 1e-6, None)
    cropped = accum[pad : pad + h, pad : pad + w]
    cropped_difficulty = difficulty_accum[pad : pad + h, pad : pad + w]
    cropped_difficulty_coverage = difficulty_weight_sum[pad : pad + h, pad : pad + w] > 0
    return _Prediction(
        mask=np.argmax(cropped, axis=-1).astype(np.uint8),
        difficulty_map=np.clip(cropped_difficulty, 0.0, 1.0),
        difficulty_coverage=cropped_difficulty_coverage,
        tile_probabilities=tile_probabilities,
        tile_centers=tile_centers,
    )


def _load_model():
    global _MODEL, _DEVICE

    if _MODEL is None:
        if not INFERENCE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Inference checkpoint not found: {INFERENCE_MODEL_PATH}. "
                "Place recommended.pth under model-artifacts/ml-days-2/global-context-segformer/."
            )

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GlobalContextSegformer()
        checkpoint = torch.load(INFERENCE_MODEL_PATH, map_location=_DEVICE, weights_only=False)
        required_keys = {
            "local_model_state_dict",
            "global_model_state_dict",
            "context_adapter_state_dict",
            "classification_head_state_dict",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise KeyError(f"Inference checkpoint is missing keys: {sorted(missing_keys)}")
        model.local_model.load_state_dict(checkpoint["local_model_state_dict"])
        model.global_model.load_state_dict(checkpoint["global_model_state_dict"])
        model.context_adapter.load_state_dict(checkpoint["context_adapter_state_dict"])
        model.classification_head.load_state_dict(checkpoint["classification_head_state_dict"])
        model.to(_DEVICE).eval()
        if _DEVICE.type == "cuda":
            model.half()
        _MODEL = model

    return _MODEL, _DEVICE


@torch.inference_mode()
def _global_feature(model: GlobalContextSegformer, device: torch.device, image: np.ndarray) -> torch.Tensor:
    tensor = _preprocess_global_image(image).to(device)
    if device.type == "cuda":
        tensor = tensor.half()
    return model.global_model.encoder(tensor)[-1]


@torch.inference_mode()
def _run_batch(
    model: GlobalContextSegformer,
    device: torch.device,
    batch: np.ndarray,
    global_feature: torch.Tensor,
    rois: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = _preprocess_batch(batch).to(device)
    roi_tensor = torch.from_numpy(rois).to(device)
    if device.type == "cuda":
        tensor = tensor.half()
    local_features = list(model.local_model.encoder(tensor))
    expanded_global_feature = global_feature.expand(tensor.shape[0], -1, -1, -1)
    local_features[-1] = model.context_adapter(local_features[-1], expanded_global_feature, roi_tensor)
    decoder_output = model.local_model.decoder(local_features)
    segmentation_logits = model.local_model.segmentation_head(decoder_output)
    classification_logits = model.classification_head(local_features[-1])
    segmentation_probabilities = F.softmax(segmentation_logits.float(), dim=1)
    difficulty_probabilities = torch.sigmoid(classification_logits.float())
    return (
        segmentation_probabilities.permute(0, 2, 3, 1).cpu().numpy(),
        difficulty_probabilities.cpu().numpy(),
    )


def _preprocess_batch(batch: np.ndarray) -> torch.Tensor:
    tensor = batch.astype(np.float32) / 255.0
    tensor = (tensor - GRAY_MEAN) / GRAY_STD
    return torch.from_numpy(tensor[:, None, :, :]).contiguous()


def _preprocess_global_image(image: np.ndarray) -> torch.Tensor:
    height, width = image.shape[:2]
    scale = TILE_SIZE / max(height, width)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    y0 = (TILE_SIZE - resized.shape[0]) // 2
    x0 = (TILE_SIZE - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return _preprocess_batch(canvas[None, ...])


def _tile_intersects_predicted_ore(
    probabilities: np.ndarray,
    y: int,
    x: int,
    pad: int,
    image_height: int,
    image_width: int,
) -> bool:
    tile_height, tile_width = probabilities.shape[:2]
    image_y0 = max(y - pad, 0)
    image_y1 = min(y + tile_height - pad, image_height)
    image_x0 = max(x - pad, 0)
    image_x1 = min(x + tile_width - pad, image_width)
    if image_y0 >= image_y1 or image_x0 >= image_x1:
        return False

    tile_y0 = image_y0 + pad - y
    tile_y1 = image_y1 + pad - y
    tile_x0 = image_x0 + pad - x
    tile_x1 = image_x1 + pad - x
    predicted_classes = np.argmax(probabilities[tile_y0:tile_y1, tile_x0:tile_x1], axis=-1)
    return bool(np.any(predicted_classes == ORE_CLASS_INDEX))


def _to_normalized_gray_image(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    low, high = np.percentile(lightness, (1, 99))
    if high > low:
        lightness = np.clip((lightness.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    normalized = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    return cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)


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
    difficulty_coverage: np.ndarray,
    tile_probabilities: list[float],
    tile_centers: list[tuple[int, int]],
) -> np.ndarray:
    heat_colors = _difficulty_colormap(difficulty_map)
    blended = (image.astype(np.float32) * 0.45 + heat_colors.astype(np.float32) * 0.55).astype(np.uint8)
    overlay = image.copy()
    overlay[difficulty_coverage] = blended[difficulty_coverage]

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
