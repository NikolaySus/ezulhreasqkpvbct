import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/data")).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "segmentation")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

ANNOTATION_DATA_DIR = Path(os.getenv("ANNOTATION_DATA_DIR", "/annotation-data")).resolve()
ANNOTATION_EDITING_ENABLED = os.getenv("ANNOTATION_EDITING_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
