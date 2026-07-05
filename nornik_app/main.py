from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from PIL import Image

from .artifacts import color_artifact_available, color_artifact_key, load_color_artifact, save_color_artifact
from .data import get_image_pair, list_image_pairs
from .export_jobs import ExportJobManager
from .image_settings import delete_image_settings, load_image_settings, save_image_settings
from .jobs import JobCancelled, JobFailed, JobManager, JobTimedOut
from .segmentation import (
    _approach2_class_map_with_illumination_preview,
    _closed_geometry_masks,
    approach2_defaults,
    build_sketch_geometry,
    extract_blue_contours_png,
    image_to_png_bytes,
    make_approach2_result,
    max_sketch_erosion_radius,
    parse_approach1,
    parse_approach2,
    parse_correction_settings,
    parse_sketch_settings,
    correction_defaults,
    render_edited_sketch_png,
    run_approach1,
    run_approach2,
    run_corrected_approach2,
    run_corrected_approach2_from_class_map,
    sketch_control_in_bounds,
    sketch_defaults,
)
from .sketch_edits import add_segment_to_edit, delete_segment_from_edit, load_sketch_edit, set_region_mode


app = FastAPI(title="Nornik talc segmentation demo")
templates = Jinja2Templates(directory="nornik_app/templates")
job_manager = JobManager()
export_job_manager = ExportJobManager()


def annotation_editing_enabled() -> bool:
    return os.getenv("ANNOTATION_EDITING_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def _ensure_editing_enabled() -> None:
    if not annotation_editing_enabled():
        raise HTTPException(status_code=423, detail="Annotation editing is locked")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    pairs = list_image_pairs()
    if not pairs:
        raise HTTPException(status_code=404, detail="Image pairs were not found")

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "images": [pair.name for pair in pairs],
            "default_image": next((pair.name for pair in pairs if pair.name == "2550374-2 10х.JPG"), pairs[0].name),
            "approach2_defaults": approach2_defaults(),
            "sketch_defaults": sketch_defaults(),
            "correction_defaults": correction_defaults(),
            "base_path": request.scope.get("root_path", ""),
            "editing_enabled": annotation_editing_enabled(),
        },
    )


@app.get("/api/images")
def images() -> dict[str, list[str]]:
    return {"images": [pair.name for pair in list_image_pairs()]}


@app.get("/api/image-settings/{image_name:path}")
def image_settings(image_name: str) -> dict[str, object]:
    try:
        return load_image_settings(image_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}") from exc


@app.put("/api/image-settings/{image_name:path}")
def put_image_settings(image_name: str, payload: dict[str, object] = Body(...)) -> dict[str, object]:
    _ensure_editing_enabled()
    try:
        return save_image_settings(image_name, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}") from exc


@app.delete("/api/image-settings/{image_name:path}")
def reset_image_settings(image_name: str) -> dict[str, object]:
    _ensure_editing_enabled()
    try:
        return delete_image_settings(image_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}") from exc


@app.post("/api/export/{kind}")
def start_export(kind: str) -> dict[str, object]:
    if kind not in {
        "color-masks",
        "color-previews",
        "corrected-masks",
        "corrected-previews",
        "corrected-originals",
        "all",
    }:
        raise HTTPException(status_code=404, detail=f"Unknown export kind: {kind}")
    return export_job_manager.start(kind)  # type: ignore[arg-type]


@app.get("/api/export-jobs/{job_id}")
def export_job_status(job_id: str) -> dict[str, object]:
    job = export_job_manager.status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Export job not found: {job_id}")
    return job


@app.get("/api/export-jobs/{job_id}/download")
def download_export_job(job_id: str) -> FileResponse:
    path = export_job_manager.path(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Export archive is not ready: {job_id}")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.get("/api/result/{approach}/{image_name:path}")
def result(approach: int, image_name: str, request: Request) -> dict[str, object]:
    pair = _pair_or_404(image_name)
    params = dict(request.query_params)
    job_id = params.pop("job_id", None)
    if job_id and approach == 2:
        return _run_cancelable_job(
            str(job_id),
            _result_worker,
            approach,
            image_name,
            str(pair.source_path),
            params,
        )

    with Image.open(pair.source_path) as image:
        if approach == 1:
            segmentation = run_approach1(image, parse_approach1(params))
        elif approach == 2:
            settings = parse_approach2(params)
            rgb, class_map, illumination_applied, illumination_preview_rgb = _approach2_class_map_with_illumination_preview(
                image,
                settings,
            )
            segmentation = make_approach2_result(rgb, class_map, settings)
            artifact_key = color_artifact_key(image_name, pair.source_path, settings)
            save_color_artifact(artifact_key, rgb, class_map, settings)
        else:
            raise HTTPException(status_code=404, detail="Unknown approach")

    response = _segmentation_response(segmentation)
    if approach == 2:
        response["colorArtifactKey"] = artifact_key
        response["illuminationCorrectionApplied"] = illumination_applied
        if illumination_preview_rgb is not None:
            response["illuminationPreviewImage"] = _rgb_data_url(illumination_preview_rgb)
    return response


@app.get("/api/corrected-result/{image_name:path}")
def corrected_result(image_name: str, request: Request) -> dict[str, object]:
    pair = _pair_or_404(image_name)
    params = dict(request.query_params)
    job_id = params.pop("job_id", None)
    artifact_key = params.pop("color_artifact_key", None)
    approach_settings = parse_approach2(params)
    if not artifact_key or not color_artifact_available(str(artifact_key), approach_settings):
        raise HTTPException(status_code=409, detail="Valid color_artifact_key is required")
    edit = load_sketch_edit(image_name)
    if job_id:
        return _run_cancelable_job(
            str(job_id),
            _corrected_result_worker,
            str(pair.source_path),
            str(pair.sketch_path),
            params,
            str(artifact_key),
            edit["segments"],
            edit["regionMode"],
        )

    with Image.open(pair.source_path) as image, Image.open(pair.sketch_path) as sketch:
        artifact = load_color_artifact(str(artifact_key), approach_settings)
        if artifact is None:
            raise HTTPException(status_code=409, detail="Valid color_artifact_key is required")
        segmentation = run_corrected_approach2_from_class_map(
            image,
            sketch,
            artifact.rgb,
            artifact.class_map,
            artifact.settings,
            parse_sketch_settings(params),
            edit["segments"],
            edit["regionMode"],
            parse_correction_settings(params),
        )

    encoded = base64.b64encode(segmentation.overlay_png).decode("ascii")
    return {
        "image": f"data:image/png;base64,{encoded}",
        "stats": segmentation.stats,
        "settings": segmentation.settings,
        "correctionApplied": segmentation.correction_applied,
        "correctionDetails": segmentation.correction_details,
        "cropStats": segmentation.crop_stats,
        "fullStats": segmentation.full_stats,
    }


@app.get("/api/correction/max-sketch-erosion/{image_name:path}")
def max_correction_sketch_erosion(image_name: str, request: Request) -> dict[str, object]:
    geometry = _sketch_geometry_response(image_name, dict(request.query_params))
    masks = _closed_geometry_masks(geometry)
    if masks is None:
        return {"max": 0, "applied": False}
    _crop_box, interior_mask, _boundary_mask = masks
    return {"max": max_sketch_erosion_radius(interior_mask), "applied": True}


@app.post("/api/cancel/{job_id}")
def cancel_job(job_id: str) -> dict[str, object]:
    return {"cancelled": job_manager.cancel(job_id)}


@app.get("/media/source/{image_name:path}")
def source_image(image_name: str) -> Response:
    pair = _pair_or_404(image_name)
    with Image.open(pair.source_path) as image:
        return Response(image_to_png_bytes(image), media_type="image/png")


@app.get("/media/sketch/{image_name:path}")
def sketch_image(image_name: str, request: Request) -> Response:
    pair = _pair_or_404(image_name)
    with Image.open(pair.source_path) as source, Image.open(pair.sketch_path) as sketch:
        settings = parse_sketch_settings(dict(request.query_params))
        return Response(extract_blue_contours_png(source, sketch, settings), media_type="image/png")


@app.get("/media/edited-sketch/{image_name:path}")
def edited_sketch_image(image_name: str, request: Request) -> Response:
    pair = _pair_or_404(image_name)
    settings = parse_sketch_settings(dict(request.query_params))
    edit = load_sketch_edit(image_name)
    filename = f"{Path(image_name).stem}-edited-sketch.png"
    with Image.open(pair.source_path) as source, Image.open(pair.sketch_path) as sketch:
        png = render_edited_sketch_png(source, sketch, settings, edit["segments"])
    return Response(
        png,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@app.get("/api/sketch/{image_name:path}")
def sketch_geometry(image_name: str, request: Request) -> dict[str, object]:
    return _sketch_geometry_response(image_name, dict(request.query_params))


@app.post("/api/sketch-segments/{image_name:path}")
def create_sketch_segment(
    image_name: str,
    request: Request,
    payload: dict[str, object] = Body(...),
) -> dict[str, object]:
    _ensure_editing_enabled()
    params = dict(request.query_params)
    geometry = _sketch_geometry_response(image_name, params)
    endpoints = {endpoint["id"]: endpoint for endpoint in geometry["endpoints"]}
    a_endpoint = endpoints.get(payload.get("aEndpointId", ""))
    b_endpoint = endpoints.get(payload.get("bEndpointId", ""))
    if not a_endpoint or not b_endpoint:
        raise HTTPException(status_code=400, detail="Unknown endpoint id")
    if a_endpoint["id"] == b_endpoint["id"]:
        raise HTTPException(status_code=400, detail="Segment endpoints must be different")
    candidate_id = "--".join(sorted((str(a_endpoint["id"]), str(b_endpoint["id"]))))
    candidate = next((item for item in geometry["candidates"] if item["id"] == candidate_id), None)
    control = _control_point(payload.get("control"))
    if not control:
        raise HTTPException(status_code=400, detail="Missing segment control point")
    if not candidate or not sketch_control_in_bounds(control, candidate["controlBounds"]):
        raise HTTPException(status_code=400, detail="Control point is outside the segment bounds")
    tangent = _control_point(payload.get("tangent"))
    if not tangent:
        raise HTTPException(status_code=400, detail="Missing segment tangent")

    add_segment_to_edit(
        image_name,
        str(a_endpoint["id"]),
        str(b_endpoint["id"]),
        (float(a_endpoint["x"]), float(a_endpoint["y"])),
        (float(b_endpoint["x"]), float(b_endpoint["y"])),
        control,
        tangent,
    )
    return _sketch_geometry_response(image_name, params)


@app.delete("/api/sketch-segments/{image_name:path}")
def delete_sketch_segment(image_name: str, segment_id: str, request: Request) -> dict[str, object]:
    _ensure_editing_enabled()
    delete_segment_from_edit(image_name, segment_id)
    return _sketch_geometry_response(image_name, dict(request.query_params))


@app.put("/api/sketch-region-mode/{image_name:path}")
def put_sketch_region_mode(
    image_name: str,
    request: Request,
    payload: dict[str, object] = Body(...),
) -> dict[str, object]:
    _ensure_editing_enabled()
    mode = payload.get("regionMode")
    if mode not in {"inside", "outside"}:
        raise HTTPException(status_code=400, detail="Unknown region mode")
    set_region_mode(image_name, str(mode))
    return _sketch_geometry_response(image_name, dict(request.query_params))


def _pair_or_404(image_name: str):
    try:
        return get_image_pair(image_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}") from exc


def _sketch_geometry_response(image_name: str, params: dict[str, str]) -> dict[str, object]:
    pair = _pair_or_404(image_name)
    settings = parse_sketch_settings(params)
    edit = load_sketch_edit(image_name)
    with Image.open(pair.source_path) as source, Image.open(pair.sketch_path) as sketch:
        return build_sketch_geometry(source, sketch, settings, edit["segments"], edit["regionMode"])


def _control_point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _run_cancelable_job(job_id: str, worker, *args) -> dict[str, object]:
    try:
        return job_manager.run(job_id, worker, *args)
    except JobCancelled as exc:
        raise HTTPException(status_code=499, detail="Job cancelled") from exc
    except JobTimedOut as exc:
        raise HTTPException(status_code=504, detail="Job timed out") from exc
    except JobFailed as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _result_worker(approach: int, image_name: str, source_path: str, params: dict[str, str]) -> dict[str, object]:
    with Image.open(source_path) as image:
        if approach == 2:
            settings = parse_approach2(params)
            rgb, class_map, illumination_applied, illumination_preview_rgb = _approach2_class_map_with_illumination_preview(
                image,
                settings,
            )
            segmentation = make_approach2_result(rgb, class_map, settings)
            artifact_key = color_artifact_key(image_name, Path(source_path), settings)
            save_color_artifact(artifact_key, rgb, class_map, settings)
        elif approach == 1:
            segmentation = run_approach1(image, parse_approach1(params))
            artifact_key = None
            illumination_applied = False
            illumination_preview_rgb = None
        else:
            raise ValueError(f"Unknown approach: {approach}")
    response = _segmentation_response(segmentation)
    if artifact_key:
        response["colorArtifactKey"] = artifact_key
        response["illuminationCorrectionApplied"] = illumination_applied
        if illumination_preview_rgb is not None:
            response["illuminationPreviewImage"] = _rgb_data_url(illumination_preview_rgb)
    return response


def _corrected_result_worker(
    source_path: str,
    sketch_path: str,
    params: dict[str, str],
    artifact_key: str,
    segments: list[dict[str, object]],
    region_mode: str,
) -> dict[str, object]:
    approach_settings = parse_approach2(params)
    artifact = load_color_artifact(artifact_key, approach_settings)
    if artifact is None:
        raise ValueError("Valid color_artifact_key is required")
    with Image.open(source_path) as image, Image.open(sketch_path) as sketch:
        segmentation = run_corrected_approach2_from_class_map(
            image,
            sketch,
            artifact.rgb,
            artifact.class_map,
            artifact.settings,
            parse_sketch_settings(params),
            segments,
            region_mode,
            parse_correction_settings(params),
        )

    response = _segmentation_response(segmentation)
    response.update(
        {
            "correctionApplied": segmentation.correction_applied,
            "correctionDetails": segmentation.correction_details,
            "cropStats": segmentation.crop_stats,
            "fullStats": segmentation.full_stats,
        }
    )
    return response


def _segmentation_response(segmentation) -> dict[str, object]:
    encoded = base64.b64encode(segmentation.overlay_png).decode("ascii")
    return {
        "image": f"data:image/png;base64,{encoded}",
        "stats": segmentation.stats,
        "settings": segmentation.settings,
    }


def _rgb_data_url(rgb) -> str:
    bio = BytesIO()
    Image.fromarray(rgb).save(bio, format="PNG", optimize=True)
    encoded = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
