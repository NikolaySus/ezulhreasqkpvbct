from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

_MODEL = None
_WEIGHTS = None
_DEVICE = None


def segment_image(upload_path: str, result_path: str) -> dict[str, str]:
    model, weights, device = _load_model()

    source = Image.open(upload_path).convert("RGB")
    original_size = source.size
    input_tensor = weights.transforms()(source).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)["out"]
        output = F.interpolate(output, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
        mask = output.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    overlay = _build_overlay(source, mask)
    result = Path(result_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(result, format="PNG")

    return {"result_path": str(result)}


def _load_model():
    global _MODEL, _WEIGHTS, _DEVICE

    if _MODEL is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _WEIGHTS = DeepLabV3_ResNet50_Weights.DEFAULT
        _MODEL = deeplabv3_resnet50(weights=_WEIGHTS).to(_DEVICE).eval()

    return _MODEL, _WEIGHTS, _DEVICE


def _build_overlay(source: Image.Image, mask: np.ndarray) -> Image.Image:
    image = np.asarray(source).astype(np.float32)
    palette = _palette()
    color_mask = palette[mask % len(palette)].astype(np.float32)

    foreground = mask != 0
    blended = image.copy()
    blended[foreground] = image[foreground] * 0.55 + color_mask[foreground] * 0.45
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _palette() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0],
            [220, 20, 60],
            [119, 11, 32],
            [0, 0, 142],
            [0, 0, 230],
            [106, 0, 228],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 70],
            [0, 0, 192],
            [250, 170, 30],
            [100, 170, 30],
            [220, 220, 0],
            [175, 116, 175],
            [250, 0, 30],
            [165, 42, 42],
            [255, 77, 255],
            [0, 226, 252],
            [182, 182, 255],
            [0, 82, 0],
            [120, 166, 157],
            [110, 76, 0],
        ],
        dtype=np.uint8,
    )
