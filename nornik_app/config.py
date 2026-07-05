import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = Path(os.getenv("ANNOTATION_DATA_DIR", PROJECT_ROOT / "annotation-data")).resolve()
NORNIK_ROOT = ANNOTATION_ROOT / "nornik"
DATA_ROOT = NORNIK_ROOT / "Фото руд по сортам. ч1" / "Оталькованные руды"
SKETCH_DIR_NAME = "Области оталькования"
SKETCH_ROOT = DATA_ROOT / SKETCH_DIR_NAME
SKETCH_EDITS_ROOT = NORNIK_ROOT / "sketch_edits"
IMAGE_SETTINGS_ROOT = NORNIK_ROOT / "image_settings"
COLOR_MARKUP_MASKS_ROOT = NORNIK_ROOT / "color_markup_masks"
COLOR_MARKUP_PREVIEWS_ROOT = NORNIK_ROOT / "color_markup_previews"
EXPORTS_ROOT = NORNIK_ROOT / "exports"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
