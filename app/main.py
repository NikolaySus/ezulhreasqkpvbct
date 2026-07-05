from pathlib import Path
from uuid import uuid4

import redis
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from nornik_app.data import list_image_pairs
from nornik_app.main import app as nornik_app
from rq.job import Job

app = FastAPI(title="MVP Backend")
app.mount("/api/annotation/nornik", nornik_app)
templates = Jinja2Templates(directory="app/templates")

from app.queue import QUEUE_NAME, get_queue, get_redis
from app.settings import (
    ANNOTATION_DATA_DIR,
    ANNOTATION_EDITING_ENABLED,
    DATA_DIR,
    MAX_UPLOAD_BYTES,
    RESULTS_DIR,
    UPLOADS_DIR,
)
from app.tasks import segment_image

ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tif": ".tif",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}
ALLOWED_FILENAME_EXTENSIONS = {
    ".jpeg": ".jpg",
    ".jpg": ".jpg",
    ".png": ".png",
    ".tif": ".tif",
    ".tiff": ".tiff",
    ".webp": ".webp",
}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
async def api_health() -> dict[str, object]:
    redis_ok = False
    try:
        redis_ok = bool(get_redis().ping())
    except redis.RedisError:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "service": "local-backend",
        "redis": redis_ok,
        "queue": QUEUE_NAME,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/api/hello")
async def api_hello() -> dict[str, str]:
    return {"message": "Hello from the local FastAPI backend"}


@app.get("/api/annotation/status")
async def annotation_status() -> dict[str, object]:
    image_count = len(list_image_pairs())
    return {
        "editing_enabled": ANNOTATION_EDITING_ENABLED,
        "mode": "unlocked" if ANNOTATION_EDITING_ENABLED else "locked",
        "data_dir": str(ANNOTATION_DATA_DIR),
        "nornik_available": image_count > 0,
        "image_count": image_count,
    }


@app.post("/api/segment", status_code=202)
async def create_segmentation_job(image: UploadFile = File(...)) -> dict[str, str]:
    extension = ALLOWED_CONTENT_TYPES.get(image.content_type or "")
    if extension is None and image.filename:
        extension = ALLOWED_FILENAME_EXTENSIONS.get(Path(image.filename).suffix.lower())
    if extension is None:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, TIFF and WebP images are supported")

    job_id = uuid4().hex
    upload_path = UPLOADS_DIR / f"{job_id}{extension}"
    result_path = RESULTS_DIR / f"{job_id}.png"

    size = 0
    with upload_path.open("wb") as target:
        while chunk := await image.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                upload_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            target.write(chunk)

    queue = get_queue()
    job = queue.enqueue(
        segment_image,
        str(upload_path),
        str(result_path),
        job_id=job_id,
        job_timeout=1800,
        result_ttl=24 * 60 * 60,
        failure_ttl=24 * 60 * 60,
    )

    return {"job_id": job.id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, object]:
    job = _load_job(job_id)
    status = job.get_status(refresh=True)
    response: dict[str, object] = {"job_id": job.id, "status": status}

    if status == "queued":
        response["queue_position"] = _queue_position(job.id)
    elif status == "started":
        response.update(_job_progress(job))
    elif status == "finished":
        response["result_url"] = f"/api/jobs/{job.id}/result"
        response.update(_job_progress(job))
        response.update(_job_analysis(job))
    elif status == "failed":
        response["error"] = _format_error(job)

    return response


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> FileResponse:
    job = _load_job(job_id)
    if job.get_status(refresh=True) != "finished":
        raise HTTPException(status_code=409, detail="Segmentation job is not finished")

    result_path = _result_path_from_job(job)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Segmentation result is missing")

    return FileResponse(result_path, media_type="image/png", filename=f"{job.id}-overlay.png")


def _load_job(job_id: str) -> Job:
    try:
        return Job.fetch(job_id, connection=get_redis())
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


def _queue_position(job_id: str) -> int | None:
    job_ids = get_queue().job_ids
    try:
        return job_ids.index(job_id) + 1
    except ValueError:
        return None


def _result_path_from_job(job: Job) -> Path:
    if isinstance(job.result, dict) and isinstance(job.result.get("result_path"), str):
        result_path = Path(job.result["result_path"])
    else:
        result_path = RESULTS_DIR / f"{job.id}.png"

    if DATA_DIR not in result_path.resolve().parents and result_path.resolve() != DATA_DIR:
        raise HTTPException(status_code=400, detail="Invalid result path")
    return result_path


def _job_analysis(job: Job) -> dict[str, object]:
    try:
        result = job.return_value(refresh=True)
    except Exception:
        return {}

    if not isinstance(result, dict):
        return {}

    analysis: dict[str, object] = {}
    talc_ratio = result.get("talc_ratio")
    if isinstance(talc_ratio, int | float):
        analysis["talc_ratio"] = float(talc_ratio)
        analysis["is_talcose"] = bool(result.get("is_talcose"))

    classes = result.get("classes")
    if isinstance(classes, list | tuple):
        analysis["classes"] = list(classes)

    return {"analysis": analysis} if analysis else {}


def _job_progress(job: Job) -> dict[str, object]:
    progress = job.meta.get("progress")
    if not isinstance(progress, dict):
        return {}

    response: dict[str, object] = {}
    stage = progress.get("stage")
    if isinstance(stage, str):
        response["stage"] = stage

    for key in ("processed_tiles", "total_tiles"):
        value = progress.get(key)
        if isinstance(value, int):
            response[key] = value

    percent = progress.get("percent")
    if isinstance(percent, int | float):
        response["percent"] = float(percent)

    return {"progress": response} if response else {}


def _format_error(job: Job) -> str:
    if not job.exc_info:
        return "Segmentation job failed"
    return job.exc_info.strip().splitlines()[-1][-500:]
