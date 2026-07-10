import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "segmentation")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
INFERENCE_MODEL_PATH = Path(
    os.getenv("INFERENCE_MODEL_PATH", "/model-artifacts/ml-days-2/global-context-segformer/recommended.pth")
).resolve()
SEGMENTATION_MODEL_PATH = INFERENCE_MODEL_PATH
ORE_TILE_RATIO_EXCLUSION_FACTOR = float(os.getenv("ORE_TILE_RATIO_EXCLUSION_FACTOR", "10"))

ANNOTATION_DATA_DIR = Path(os.getenv("ANNOTATION_DATA_DIR", "/annotation-data")).resolve()
ANNOTATION_EDITING_ENABLED = os.getenv("ANNOTATION_EDITING_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
