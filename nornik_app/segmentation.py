from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from io import BytesIO
from math import ceil, cos, pi, radians, sin, tau
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import color, morphology, segmentation
from sklearn.cluster import HDBSCAN, KMeans


logger = logging.getLogger(__name__)
ClassName = Literal["ore", "matrix", "talc", "damage"]
CLASS_NAMES: tuple[ClassName, ...] = ("ore", "matrix", "talc", "damage")


@dataclass(frozen=True)
class Approach1Settings:
    k: int = 7
    ore_l_min: float = 56.0
    ore_b_min: float = 8.0
    talc_l_max: float = 30.0
    min_component_area: int = 70
    morph_radius: int = 2
    max_work_side: int = 1300
    sample_pixels: int = 90000


@dataclass(frozen=True)
class Approach2Settings:
    matrix_subclusters: int = 5
    ore_boundary_shift: float = 0.72
    talc_dark_quantile: float = 0.1
    min_component_area: int = 10
    morph_radius: int = 1
    edge_refinement_strength: float = 1.0
    edge_gradient_radius: int = 1
    edge_seed_erosion_radius: int = 2
    edge_transition_radius: int = 3
    edge_soft_delta_e: float = 6.5
    edge_min_region_area: int = 100
    illumination_correction_strength: float = 0.8
    illumination_ore_erosion_radius: int = 12
    illumination_min_ore_area: int = 5000
    illumination_outlier_quantile: float = 0.1
    illumination_max_delta_l: float = 20.0
    local_shadow_correction_strength: float = 1.0
    local_shadow_radius: int = 66
    local_shadow_max_delta_l: float = 20.0
    local_shadow_chroma_barrier: float = 1.0
    local_shadow_lightness_barrier: float = 0.2
    max_work_side: int = 1300
    sample_pixels: int = 90000


@dataclass(frozen=True)
class SegmentationResult:
    overlay_png: bytes
    stats: dict[str, float]
    settings: dict[str, int | float]


@dataclass(frozen=True)
class CorrectedSegmentationResult(SegmentationResult):
    correction_applied: bool
    correction_details: dict[str, Any]
    crop_stats: dict[str, float]
    full_stats: dict[str, float]


@dataclass(frozen=True)
class SketchSettings:
    endpoint_min_branch_length: int = 40
    connection_max_distance: float = 300.0
    connection_neighbors_per_endpoint: int = 4
    angular_neighbors_per_endpoint: int = 2


@dataclass(frozen=True)
class CorrectionSettings:
    crop_excluded_talc_max_fraction: float = 0.125
    crop_min_area_ratio: float = 0.50
    crop_max_rotation_degrees: int = 0
    crop_sketch_erosion_radius: int = 42


OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    "ore": (0, 245, 85),
    "matrix": (118, 126, 128),
    "talc": (0, 170, 255),
    "damage": (185, 0, 70),
}
CORRECTED_OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {
    **OVERLAY_COLORS,
    "ore": (255, 40, 40),
}


def approach1_defaults() -> Approach1Settings:
    return Approach1Settings()


def approach2_defaults() -> Approach2Settings:
    return Approach2Settings()


def sketch_defaults() -> SketchSettings:
    return SketchSettings()


def correction_defaults() -> CorrectionSettings:
    return CorrectionSettings()


def parse_approach1(params: dict[str, str]) -> Approach1Settings:
    defaults = approach1_defaults()
    return Approach1Settings(
        k=_clamp_int(params.get("k"), defaults.k, 3, 12),
        ore_l_min=_clamp_float(params.get("ore_l_min"), defaults.ore_l_min, 0, 100),
        ore_b_min=_clamp_float(params.get("ore_b_min"), defaults.ore_b_min, -30, 60),
        talc_l_max=_clamp_float(params.get("talc_l_max"), defaults.talc_l_max, 0, 80),
        min_component_area=_clamp_int(params.get("min_component_area"), defaults.min_component_area, 0, 5000),
        morph_radius=_clamp_int(params.get("morph_radius"), defaults.morph_radius, 0, 8),
        max_work_side=_clamp_int(params.get("max_work_side"), defaults.max_work_side, 500, 2200),
        sample_pixels=_clamp_int(params.get("sample_pixels"), defaults.sample_pixels, 10000, 300000),
    )


def parse_approach2(params: dict[str, str]) -> Approach2Settings:
    defaults = approach2_defaults()
    return Approach2Settings(
        matrix_subclusters=_clamp_int(params.get("matrix_subclusters"), defaults.matrix_subclusters, 2, 5),
        ore_boundary_shift=_clamp_float(
            params.get("ore_boundary_shift"),
            defaults.ore_boundary_shift,
            -1.0,
            1.0,
        ),
        talc_dark_quantile=_clamp_float(params.get("talc_dark_quantile"), defaults.talc_dark_quantile, 0.05, 0.8),
        min_component_area=_clamp_int(params.get("min_component_area"), defaults.min_component_area, 0, 5000),
        morph_radius=_clamp_int(params.get("morph_radius"), defaults.morph_radius, 0, 8),
        edge_refinement_strength=_clamp_float(
            params.get("edge_refinement_strength"),
            defaults.edge_refinement_strength,
            0.0,
            1.0,
        ),
        edge_gradient_radius=_clamp_int(params.get("edge_gradient_radius"), defaults.edge_gradient_radius, 1, 8),
        edge_seed_erosion_radius=_clamp_int(
            params.get("edge_seed_erosion_radius"),
            defaults.edge_seed_erosion_radius,
            0,
            8,
        ),
        edge_transition_radius=_clamp_int(params.get("edge_transition_radius"), defaults.edge_transition_radius, 0, 12),
        edge_soft_delta_e=_clamp_float(params.get("edge_soft_delta_e"), defaults.edge_soft_delta_e, 1.0, 40.0),
        edge_min_region_area=_clamp_int(params.get("edge_min_region_area"), defaults.edge_min_region_area, 0, 5000),
        illumination_correction_strength=_clamp_float(
            params.get("illumination_correction_strength"),
            defaults.illumination_correction_strength,
            0.0,
            1.0,
        ),
        illumination_ore_erosion_radius=_clamp_int(
            params.get("illumination_ore_erosion_radius"),
            defaults.illumination_ore_erosion_radius,
            0,
            12,
        ),
        illumination_min_ore_area=_clamp_int(
            params.get("illumination_min_ore_area"),
            defaults.illumination_min_ore_area,
            0,
            100000,
        ),
        illumination_outlier_quantile=_clamp_float(
            params.get("illumination_outlier_quantile"),
            defaults.illumination_outlier_quantile,
            0.0,
            0.35,
        ),
        illumination_max_delta_l=_clamp_float(
            params.get("illumination_max_delta_l"),
            defaults.illumination_max_delta_l,
            0.0,
            50.0,
        ),
        local_shadow_correction_strength=_clamp_float(
            params.get("local_shadow_correction_strength"),
            defaults.local_shadow_correction_strength,
            0.0,
            1.0,
        ),
        local_shadow_radius=_clamp_int(params.get("local_shadow_radius"), defaults.local_shadow_radius, 3, 120),
        local_shadow_max_delta_l=_clamp_float(
            params.get("local_shadow_max_delta_l"),
            defaults.local_shadow_max_delta_l,
            0.0,
            40.0,
        ),
        local_shadow_chroma_barrier=_clamp_float(
            params.get("local_shadow_chroma_barrier"),
            defaults.local_shadow_chroma_barrier,
            0.0,
            3.0,
        ),
        local_shadow_lightness_barrier=_clamp_float(
            params.get("local_shadow_lightness_barrier"),
            defaults.local_shadow_lightness_barrier,
            0.0,
            2.0,
        ),
        max_work_side=_clamp_int(params.get("max_work_side"), defaults.max_work_side, 500, 2200),
        sample_pixels=_clamp_int(params.get("sample_pixels"), defaults.sample_pixels, 10000, 300000),
    )


def parse_sketch_settings(params: dict[str, str]) -> SketchSettings:
    defaults = sketch_defaults()
    return SketchSettings(
        endpoint_min_branch_length=_clamp_int(
            params.get("endpoint_min_branch_length"),
            defaults.endpoint_min_branch_length,
            0,
            200,
        ),
        connection_max_distance=_clamp_float(
            params.get("connection_max_distance"),
            defaults.connection_max_distance,
            0,
            3000,
        ),
        connection_neighbors_per_endpoint=_clamp_int(
            params.get("connection_neighbors_per_endpoint"),
            defaults.connection_neighbors_per_endpoint,
            1,
            12,
        ),
        angular_neighbors_per_endpoint=_clamp_int(
            params.get("angular_neighbors_per_endpoint"),
            defaults.angular_neighbors_per_endpoint,
            0,
            4,
        ),
    )


def parse_correction_settings(params: dict[str, str]) -> CorrectionSettings:
    defaults = correction_defaults()
    return CorrectionSettings(
        crop_excluded_talc_max_fraction=_clamp_float(
            params.get("crop_excluded_talc_max_fraction"),
            defaults.crop_excluded_talc_max_fraction,
            0.0,
            1.0,
        ),
        crop_min_area_ratio=_clamp_float(
            params.get("crop_min_area_ratio"),
            defaults.crop_min_area_ratio,
            0.1,
            1.0,
        ),
        crop_max_rotation_degrees=0,
        crop_sketch_erosion_radius=_clamp_int(
            params.get("crop_sketch_erosion_radius"),
            defaults.crop_sketch_erosion_radius,
            0,
            5000,
        ),
    )


def run_approach1(image: Image.Image, settings: Approach1Settings) -> SegmentationResult:
    rgb = _prepare_rgb(image, settings.max_work_side)
    lab = color.rgb2lab(rgb / 255.0).astype(np.float32)
    pixels = lab.reshape(-1, 3)
    labels, centers = _kmeans_labels(pixels, settings.k, settings.sample_pixels)

    class_by_cluster: list[ClassName] = []
    for center in centers:
        l_value, _a_value, b_value = center
        if l_value >= settings.ore_l_min and b_value >= settings.ore_b_min:
            class_by_cluster.append("ore")
        elif l_value <= settings.talc_l_max:
            class_by_cluster.append("talc")
        else:
            class_by_cluster.append("matrix")

    class_map = _labels_to_class_map(labels.reshape(lab.shape[:2]), class_by_cluster)
    class_map = _postprocess(class_map, settings.min_component_area, settings.morph_radius)
    return _make_result(rgb, class_map, asdict(settings))


def run_approach2(image: Image.Image, settings: Approach2Settings) -> SegmentationResult:
    rgb, class_map = _approach2_class_map(image, settings)
    return make_approach2_result(rgb, class_map, settings)


def make_approach2_result(
    rgb: np.ndarray,
    class_map: np.ndarray,
    settings: Approach2Settings,
) -> SegmentationResult:
    return _make_result(rgb, class_map, asdict(settings))


def run_corrected_approach2(
    image: Image.Image,
    sketch: Image.Image,
    approach_settings: Approach2Settings,
    sketch_settings: SketchSettings,
    saved_segments: list[dict[str, Any]] | None = None,
    region_mode: str = "inside",
    correction_settings: CorrectionSettings | None = None,
) -> CorrectedSegmentationResult:
    rgb, class_map = _approach2_class_map(image, approach_settings)
    return run_corrected_approach2_from_class_map(
        image,
        sketch,
        rgb,
        class_map,
        approach_settings,
        sketch_settings,
        saved_segments,
        region_mode,
        correction_settings,
    )


def run_corrected_approach2_from_class_map(
    image: Image.Image,
    sketch: Image.Image,
    rgb: np.ndarray,
    class_map: np.ndarray,
    approach_settings: Approach2Settings,
    sketch_settings: SketchSettings,
    saved_segments: list[dict[str, Any]] | None = None,
    region_mode: str = "inside",
    correction_settings: CorrectionSettings | None = None,
) -> CorrectedSegmentationResult:
    corrected_data = corrected_approach2_data_from_class_map(
        image,
        sketch,
        rgb,
        class_map,
        approach_settings,
        sketch_settings,
        saved_segments,
        region_mode,
        correction_settings,
    )
    overlay = _overlay_array(rgb, corrected_data["class_map"], CORRECTED_OVERLAY_COLORS)
    corrected = (rgb.astype(np.float32) * 0.3).copy()
    crop_mask = corrected_data["crop_mask"]
    corrected[crop_mask] = overlay[crop_mask]
    bio = BytesIO()
    Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8)).save(bio, format="PNG", optimize=True)
    return CorrectedSegmentationResult(
        bio.getvalue(),
        corrected_data["full_stats"],
        asdict(approach_settings),
        bool(corrected_data["correction_applied"]),
        corrected_data["details"],
        corrected_data["crop_stats"],
        corrected_data["full_stats"],
    )


def corrected_approach2_data_from_class_map(
    image: Image.Image,
    sketch: Image.Image,
    rgb: np.ndarray,
    class_map: np.ndarray,
    approach_settings: Approach2Settings,
    sketch_settings: SketchSettings,
    saved_segments: list[dict[str, Any]] | None = None,
    region_mode: str = "inside",
    correction_settings: CorrectionSettings | None = None,
) -> dict[str, Any]:
    correction_settings = correction_settings or correction_defaults()
    base_stats = _stats(class_map)
    geometry = build_sketch_geometry(image, sketch, sketch_settings, saved_segments, region_mode)
    region = _closed_geometry_masks(geometry)
    if region is None:
        full_mask = np.ones(class_map.shape, dtype=bool)
        return {
            "class_map": np.array(class_map, copy=True),
            "crop_mask": full_mask,
            "crop_box": (0, 0, int(class_map.shape[1]), int(class_map.shape[0])),
            "crop_stats": base_stats,
            "full_stats": base_stats,
            "details": _empty_correction_details(correction_settings),
            "correction_applied": False,
        }

    crop_box, interior_mask, boundary_mask = region
    scaled_interior = _resize_bool_mask(interior_mask, rgb.shape[1], rgb.shape[0])
    scaled_boundary = _resize_bool_mask(boundary_mask, rgb.shape[1], rgb.shape[0])
    talc_selection = _select_talc_segments_with_hdbscan(rgb, class_map, scaled_interior, scaled_boundary)
    render_map = np.array(class_map, copy=True)
    render_map[(render_map == "talc") & ~talc_selection["mask"]] = "matrix"
    left, top, right, bottom = _scale_box(crop_box, geometry["width"], geometry["height"], rgb.shape[1], rgb.shape[0])
    crop_selection = _enhanced_correction_crop(
        (left, top, right, bottom),
        scaled_interior,
        class_map == "talc",
        (class_map == "talc") & ~talc_selection["mask"],
        correction_settings,
    )
    full_stats = _stats(render_map)
    crop_stats = _masked_stats(render_map, crop_selection["mask"])
    correction_details = {**talc_selection["details"], **crop_selection["details"]}
    return {
        "class_map": render_map,
        "crop_mask": crop_selection["mask"],
        "crop_box": _mask_box(crop_selection["mask"]),
        "crop_stats": crop_stats,
        "full_stats": full_stats,
        "details": correction_details,
        "correction_applied": True,
    }


def _empty_correction_details(correction_settings: CorrectionSettings) -> dict[str, Any]:
    return {
        "talcSegments": 0,
        "requiredSegments": 0,
        "centroidInsideSegments": 0,
        "intersectingSegments": 0,
        "selectedClusters": 0,
        "selectedSegments": 0,
        "fallback": False,
        "testedParameterSets": 0,
        "validParameterSets": 0,
        "minClusterSize": None,
        "minSamples": None,
        "featureSet": None,
        "supplementalSegments": 0,
        "insideRegionAreaMinimum": 0,
        "outsideAreaFilterThreshold": 0,
        "outsideAreaFilteredSegments": 0,
        "outsideAreaFilteredPixels": 0,
        "cropEnhanced": False,
        "cropFallback": False,
        "cropType": "none",
        "cropAngleDegrees": 0.0,
        "cropAreaRatio": 1.0,
        "cropSketchErosionRadius": int(correction_settings.crop_sketch_erosion_radius),
        "excludedTalcPixelsBefore": 0,
        "excludedTalcPixelsAfter": 0,
        "talcPixelsInsideCrop": 0,
        "visibleExcludedTalcFraction": 0.0,
    }


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, int(mask.shape[1]), int(mask.shape[0]))
    return (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)


def _approach2_class_map(image: Image.Image, settings: Approach2Settings) -> tuple[np.ndarray, np.ndarray]:
    rgb, class_map, _applied = _approach2_class_map_with_illumination_status(image, settings)
    return rgb, class_map


def _approach2_class_map_with_illumination_status(
    image: Image.Image,
    settings: Approach2Settings,
) -> tuple[np.ndarray, np.ndarray, bool]:
    rgb, class_map, applied, _preview_rgb = _approach2_class_map_with_illumination_preview(image, settings)
    return rgb, class_map, applied


def _approach2_class_map_with_illumination_preview(
    image: Image.Image,
    settings: Approach2Settings,
) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray | None]:
    rgb = _prepare_rgb(image, settings.max_work_side)
    lab = color.rgb2lab(rgb / 255.0).astype(np.float32)
    illumination_applied = False
    illumination_preview_rgb = None
    if settings.illumination_correction_strength > 0 or settings.local_shadow_correction_strength > 0:
        initial_class_map = _approach2_class_map_from_lab(lab, settings, apply_edge_refinement=False)
        lab, illumination_applied = _correct_lab_illumination_from_ore(lab, initial_class_map, settings)
        if illumination_applied:
            illumination_preview_rgb = _lab_to_rgb_uint8(lab)
    class_map = _approach2_class_map_from_lab(lab, settings, apply_edge_refinement=True)
    return rgb, class_map, illumination_applied, illumination_preview_rgb


def _approach2_class_map_from_lab(
    lab: np.ndarray,
    settings: Approach2Settings,
    apply_edge_refinement: bool,
) -> np.ndarray:
    pixels = lab.reshape(-1, 3)

    global_labels, global_centers = _kmeans_labels(pixels, 2, settings.sample_pixels)
    global_labels_2d = global_labels.reshape(lab.shape[:2])
    scores = global_centers[:, 0] + 0.55 * global_centers[:, 2]
    ore_cluster = int(np.argmax(scores))
    matrix_cluster = 1 - ore_cluster

    class_map = np.full(global_labels_2d.shape, "matrix", dtype=object)
    ore_center = global_centers[ore_cluster]
    matrix_center = global_centers[matrix_cluster]
    ore_distance = np.sum((lab - ore_center) ** 2, axis=2)
    matrix_distance = np.sum((lab - matrix_center) ** 2, axis=2)
    boundary_score = (matrix_distance - ore_distance) / (matrix_distance + ore_distance + 1e-6)
    ore_mask = boundary_score >= -settings.ore_boundary_shift
    class_map[ore_mask] = "ore"
    matrix_mask = ~ore_mask

    matrix_pixels = lab[matrix_mask]
    if len(matrix_pixels) >= settings.matrix_subclusters:
        sub_labels, sub_centers = _kmeans_labels(matrix_pixels, settings.matrix_subclusters, settings.sample_pixels)
        l_values = sub_centers[:, 0]
        l_threshold = float(np.quantile(l_values, settings.talc_dark_quantile))
        talc_subclusters = np.flatnonzero(l_values <= l_threshold)
        matrix_positions = np.flatnonzero(matrix_mask.reshape(-1))
        flat_map = class_map.reshape(-1)
        talc_positions = matrix_positions[np.isin(sub_labels, talc_subclusters)]
        flat_map[talc_positions] = "talc"

    if apply_edge_refinement:
        class_map = _edge_refine_talc_and_damage(lab, class_map, matrix_mask, settings)
    class_map = _postprocess(class_map, settings.min_component_area, settings.morph_radius)
    return class_map


def _prepare_rgb(image: Image.Image, max_side: int) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def _lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    rgb = color.lab2rgb(lab.astype(np.float32)).astype(np.float32)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _kmeans_labels(pixels: np.ndarray, k: int, sample_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    if len(pixels) == 0:
        raise ValueError("Image contains no pixels")

    sample_count = min(len(pixels), sample_pixels)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(pixels), size=sample_count, replace=False)
    sample = pixels[sample_idx]
    cuml_kmeans = _cuml_kmeans_class()
    if cuml_kmeans is not None:
        try:
            model = cuml_kmeans(n_clusters=min(k, len(sample)), n_init=5, random_state=42, output_type="numpy")
            model.fit(sample)
            labels = model.predict(pixels)
            return _as_numpy(labels).astype(np.int32), _as_numpy(model.cluster_centers_).astype(np.float32)
        except Exception as exc:
            logger.warning("cuML KMeans failed, falling back to sklearn: %s", exc)
    model = KMeans(n_clusters=min(k, len(sample)), n_init=5, random_state=42)
    model.fit(sample)
    labels = model.predict(pixels)
    return labels.astype(np.int32), model.cluster_centers_.astype(np.float32)


@lru_cache(maxsize=1)
def _cuml_kmeans_class():
    try:
        from cuml.cluster import KMeans as CumlKMeans
    except Exception:
        return None
    return CumlKMeans


def kmeans_backend_name() -> str:
    return "cuml" if _cuml_kmeans_class() is not None else "sklearn"


def _as_numpy(values) -> np.ndarray:
    if hasattr(values, "get"):
        return values.get()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _labels_to_class_map(labels: np.ndarray, class_by_cluster: list[ClassName]) -> np.ndarray:
    class_map = np.empty(labels.shape, dtype=object)
    for cluster_id, class_name in enumerate(class_by_cluster):
        class_map[labels == cluster_id] = class_name
    return class_map


def _correct_lab_illumination_from_ore(
    lab: np.ndarray,
    class_map: np.ndarray,
    settings: Approach2Settings,
) -> tuple[np.ndarray, bool]:
    corrected = np.array(lab, copy=True)
    applied = False
    corrected, global_applied = _apply_global_ore_illumination_correction(corrected, class_map, settings)
    applied = applied or global_applied
    corrected, local_applied = _apply_local_shadow_correction(corrected, settings)
    applied = applied or local_applied
    return corrected, applied


def _apply_global_ore_illumination_correction(
    lab: np.ndarray,
    class_map: np.ndarray,
    settings: Approach2Settings,
) -> tuple[np.ndarray, bool]:
    strength = settings.illumination_correction_strength
    if strength <= 0 or settings.illumination_max_delta_l <= 0:
        return lab, False
    ore_mask = class_map == "ore"
    if settings.min_component_area > 0:
        ore_mask = _remove_small_objects_compat(ore_mask, settings.min_component_area)
    if settings.illumination_ore_erosion_radius > 0:
        ore_mask = morphology.erosion(ore_mask, morphology.disk(settings.illumination_ore_erosion_radius))

    if int(np.count_nonzero(ore_mask)) < settings.illumination_min_ore_area:
        return lab, False

    y_coords, x_coords = np.nonzero(ore_mask)
    l_values = lab[:, :, 0][ore_mask]
    trim = settings.illumination_outlier_quantile
    if trim > 0 and len(l_values) >= 20:
        low, high = np.quantile(l_values, [trim, 1.0 - trim])
        keep = (l_values >= low) & (l_values <= high)
        y_coords = y_coords[keep]
        x_coords = x_coords[keep]
        l_values = l_values[keep]

    if len(l_values) < max(settings.illumination_min_ore_area, 6):
        return lab, False

    sample_limit = min(len(l_values), settings.sample_pixels)
    if sample_limit < len(l_values):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(l_values), size=sample_limit, replace=False)
        y_coords = y_coords[sample_idx]
        x_coords = x_coords[sample_idx]
        l_values = l_values[sample_idx]

    height, width = lab.shape[:2]
    design = _quadratic_surface_design(x_coords, y_coords, width, height)
    try:
        coefficients, *_ = np.linalg.lstsq(design, l_values.astype(np.float32), rcond=None)
    except np.linalg.LinAlgError:
        return lab, False
    if not np.all(np.isfinite(coefficients)):
        return lab, False

    yy, xx = np.indices((height, width), dtype=np.float32)
    full_design = _quadratic_surface_design(xx.reshape(-1), yy.reshape(-1), width, height)
    fitted = (full_design @ coefficients).reshape(height, width).astype(np.float32)
    if not np.all(np.isfinite(fitted)):
        return lab, False

    ore_fitted = fitted[ore_mask]
    if len(ore_fitted) == 0:
        return lab, False
    delta_l = fitted - float(np.median(ore_fitted))
    delta_l = np.clip(delta_l, -settings.illumination_max_delta_l, settings.illumination_max_delta_l)

    corrected = np.array(lab, copy=True)
    corrected[:, :, 0] = np.clip(corrected[:, :, 0] - strength * delta_l, 0.0, 100.0)
    return corrected, bool(np.any(np.abs(delta_l) > 1e-6))


def _apply_local_shadow_correction(
    lab: np.ndarray,
    settings: Approach2Settings,
) -> tuple[np.ndarray, bool]:
    strength = settings.local_shadow_correction_strength
    if strength <= 0 or settings.local_shadow_max_delta_l <= 0:
        return lab, False

    l_channel = lab[:, :, 0].astype(np.float32)
    radius = max(3, settings.local_shadow_radius)
    working_lab, scale = _local_shadow_working_lab(lab, radius)
    working_l = working_lab[:, :, 0].astype(np.float32)
    working_radius = max(3, int(round(radius * scale)))
    diameter = working_radius * 2 + 1
    sigma_color = max(4.0, settings.local_shadow_max_delta_l * 1.5)
    local_background = cv2.bilateralFilter(working_l, diameter, sigma_color, float(working_radius))
    shadow_delta = np.maximum(local_background - working_l, 0.0)
    if not np.any(shadow_delta > 1e-6):
        return lab, False

    boundary_score = _local_shadow_boundary_score(working_lab, settings, gradient_radius=max(1, int(round(settings.edge_gradient_radius * scale))))
    shadow_delta *= 1.0 - boundary_score
    shadow_delta = np.clip(shadow_delta, 0.0, settings.local_shadow_max_delta_l)
    if not np.any(shadow_delta > 1e-6):
        return lab, False
    if scale < 1.0:
        shadow_delta = cv2.resize(shadow_delta, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)

    corrected = np.array(lab, copy=True)
    corrected[:, :, 0] = np.clip(l_channel + strength * shadow_delta, 0.0, 100.0)
    return corrected, True


def _local_shadow_working_lab(lab: np.ndarray, radius: int) -> tuple[np.ndarray, float]:
    target_radius = 32
    scale = min(1.0, target_radius / float(max(radius, 1)))
    if scale >= 1.0:
        return lab, 1.0
    width = max(1, int(round(lab.shape[1] * scale)))
    height = max(1, int(round(lab.shape[0] * scale)))
    channels = [
        cv2.resize(lab[:, :, channel].astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        for channel in range(3)
    ]
    return np.dstack(channels).astype(np.float32), scale


def _local_shadow_boundary_score(
    lab: np.ndarray,
    settings: Approach2Settings,
    gradient_radius: int | None = None,
) -> np.ndarray:
    gradient_radius = max(1, gradient_radius if gradient_radius is not None else settings.edge_gradient_radius)
    gradient_l = _single_channel_morphological_gradient(lab[:, :, 0], gradient_radius)
    gradient_a = _single_channel_morphological_gradient(lab[:, :, 1], gradient_radius)
    gradient_b = _single_channel_morphological_gradient(lab[:, :, 2], gradient_radius)
    gradient_ab = np.sqrt(gradient_a**2 + gradient_b**2)

    l_norm = _robust_normalized_gradient(gradient_l)
    ab_norm = _robust_normalized_gradient(gradient_ab)
    boundary = settings.local_shadow_lightness_barrier * l_norm + settings.local_shadow_chroma_barrier * ab_norm
    boundary = np.clip(boundary, 0.0, 1.0).astype(np.float32)
    dilation_radius = max(1, min((gradient_radius + settings.local_shadow_radius // 12), 12))
    if dilation_radius > 0:
        boundary = morphology.dilation(boundary, morphology.disk(dilation_radius)).astype(np.float32)
    return np.clip(boundary, 0.0, 1.0).astype(np.float32)


def _quadratic_surface_design(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    x = (x_coords.astype(np.float32) / x_scale) * 2.0 - 1.0
    y = (y_coords.astype(np.float32) / y_scale) * 2.0 - 1.0
    return np.column_stack((x * x, y * y, x * y, x, y, np.ones_like(x))).astype(np.float32)


def _edge_refine_talc_and_damage(
    lab: np.ndarray,
    class_map: np.ndarray,
    matrix_mask: np.ndarray,
    settings: Approach2Settings,
) -> np.ndarray:
    if settings.edge_refinement_strength <= 0:
        return class_map

    talc_mask = class_map == "talc"
    if not np.any(talc_mask):
        return class_map
    if settings.edge_min_region_area > 0:
        refinement_talc_mask = _remove_small_objects_compat(talc_mask, settings.edge_min_region_area)
    else:
        refinement_talc_mask = talc_mask
    if not np.any(refinement_talc_mask):
        return class_map

    transition_radius = settings.edge_transition_radius
    gradient = _lab_morphological_gradient(lab, settings.edge_gradient_radius)
    nonzero_gradient = gradient[gradient > 1e-6]
    robust_gradient = float(np.percentile(nonzero_gradient, 95)) if len(nonzero_gradient) else 0.0
    if robust_gradient <= 1e-6:
        return class_map
    gradient_norm = np.clip(gradient / robust_gradient, 0.0, 1.0)

    watershed_mask = refinement_talc_mask.copy()
    if transition_radius > 0:
        watershed_mask = morphology.dilation(watershed_mask, morphology.disk(transition_radius))
    watershed_mask &= matrix_mask
    if not np.any(watershed_mask):
        return class_map

    markers = _talc_seed_markers(refinement_talc_mask, settings.edge_seed_erosion_radius)
    if int(markers.max()) == 0:
        return class_map

    labels = segmentation.watershed(gradient_norm, markers, mask=watershed_mask)
    refined_talc = talc_mask.copy()
    damage_mask = np.zeros(talc_mask.shape, dtype=bool)
    region_slices = ndimage.find_objects(labels)
    edge_band_disk = morphology.disk(max(1, transition_radius))
    boundary_disk = morphology.disk(1)
    pad = max(1, transition_radius) + 1

    for region_id, region_slice in enumerate(region_slices, start=1):
        if region_slice is None:
            continue
        y_slice, x_slice = region_slice
        y0 = max(0, y_slice.start - pad)
        y1 = min(labels.shape[0], y_slice.stop + pad)
        x0 = max(0, x_slice.start - pad)
        x1 = min(labels.shape[1], x_slice.stop + pad)
        local = np.s_[y0:y1, x0:x1]

        local_labels = labels[local]
        local_talc = refinement_talc_mask[local]
        local_matrix = matrix_mask[local]
        region = local_labels == region_id
        raw_region = region & local_talc
        if not np.any(raw_region):
            continue
        if settings.edge_min_region_area > 0 and int(np.count_nonzero(raw_region)) < settings.edge_min_region_area:
            continue

        outer_band = morphology.dilation(raw_region, edge_band_disk) & local_matrix & ~raw_region
        inner_band = raw_region & morphology.dilation(~raw_region, boundary_disk)
        if not np.any(inner_band):
            inner_band = raw_region
        if not np.any(outer_band):
            continue

        boundary = morphology.dilation(raw_region, boundary_disk) ^ morphology.erosion(raw_region, boundary_disk)
        boundary &= watershed_mask[local]
        if not np.any(boundary):
            boundary = raw_region

        local_lab = lab[local]
        delta_e = _mean_delta_e(local_lab, inner_band, outer_band)
        boundary_sharpness = float(np.median(gradient_norm[local][boundary]))
        sharp_delta = np.clip(delta_e / max(settings.edge_soft_delta_e, 1e-6), 0.0, 2.0) / 2.0
        damage_score = settings.edge_refinement_strength * boundary_sharpness * sharp_delta

        if damage_score >= 0.2 and boundary_sharpness >= 0.35 and delta_e >= settings.edge_soft_delta_e:
            damage_mask[local] |= raw_region
        else:
            soft_score = 1.0 - np.clip(delta_e / max(settings.edge_soft_delta_e, 1e-6), 0.0, 1.0)
            if soft_score >= 0.35:
                refined_talc[local] |= region & local_matrix

    refined = np.array(class_map, copy=True)
    refined[talc_mask] = "matrix"
    refined[refined_talc & ~damage_mask] = "talc"
    refined[damage_mask] = "damage"
    return refined


def _lab_morphological_gradient(lab: np.ndarray, radius: int) -> np.ndarray:
    gradient_sq = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel in range(3):
        gradient_sq += _single_channel_morphological_gradient(lab[:, :, channel], radius) ** 2
    return np.sqrt(gradient_sq)


def _single_channel_morphological_gradient(values: np.ndarray, radius: int) -> np.ndarray:
    kernel = morphology.disk(max(1, radius)).astype(np.uint8)
    channel_values = values.astype(np.float32)
    dilated = cv2.dilate(channel_values, kernel)
    eroded = cv2.erode(channel_values, kernel)
    return dilated - eroded


def _robust_normalized_gradient(gradient: np.ndarray) -> np.ndarray:
    nonzero = gradient[gradient > 1e-6]
    robust = float(np.percentile(nonzero, 95)) if len(nonzero) else 0.0
    if robust <= 1e-6:
        return np.zeros(gradient.shape, dtype=np.float32)
    return np.clip(gradient / robust, 0.0, 1.0).astype(np.float32)


def _talc_seed_markers(talc_mask: np.ndarray, erosion_radius: int) -> np.ndarray:
    component_count, components = cv2.connectedComponents(talc_mask.astype(np.uint8), connectivity=8)
    markers = np.zeros(talc_mask.shape, dtype=np.int32)
    marker_id = 1
    disk = morphology.disk(erosion_radius) if erosion_radius > 0 else None
    for component_id in range(1, component_count):
        component = components == component_id
        seed = morphology.erosion(component, disk) if disk is not None else component
        if not np.any(seed):
            seed = component
        seed_count, seed_components = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
        for seed_id in range(1, seed_count):
            markers[seed_components == seed_id] = marker_id
            marker_id += 1
    return markers


def _mean_delta_e(lab: np.ndarray, inner_mask: np.ndarray, outer_mask: np.ndarray) -> float:
    inner_mean = lab[inner_mask].mean(axis=0)
    outer_mean = lab[outer_mask].mean(axis=0)
    return float(np.linalg.norm(inner_mean - outer_mean))


def _postprocess(class_map: np.ndarray, min_component_area: int, morph_radius: int) -> np.ndarray:
    processed = np.array(class_map, copy=True)
    for class_name in ("ore", "talc", "damage"):
        mask = processed == class_name
        if morph_radius > 0:
            disk = morphology.disk(morph_radius)
            mask = morphology.opening(mask, disk)
            mask = morphology.closing(mask, disk)
        if min_component_area > 0:
            mask = _remove_small_objects_compat(mask, min_component_area)
        processed[processed == class_name] = "matrix"
        processed[mask] = class_name
    return processed


def _remove_small_objects_compat(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 0:
        return mask
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return morphology.remove_small_objects(mask, min_size=min_component_area)
    except TypeError:
        return morphology.remove_small_objects(mask, max_size=min_component_area - 1)


def _make_result(rgb: np.ndarray, class_map: np.ndarray, settings: dict[str, int | float]) -> SegmentationResult:
    overlay = _overlay_array(rgb, class_map)
    stats = _stats(class_map)
    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    bio = BytesIO()
    image.save(bio, format="PNG", optimize=True)
    return SegmentationResult(bio.getvalue(), stats, settings)


def _overlay_array(
    rgb: np.ndarray,
    class_map: np.ndarray,
    colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    overlay = rgb.astype(np.float32).copy()
    colors = colors or OVERLAY_COLORS
    alpha_by_class = {"ore": 0.65, "matrix": 0.12, "talc": 0.62, "damage": 0.68}
    for class_name, color_rgb in colors.items():
        mask = class_map == class_name
        if not np.any(mask):
            continue
        color_arr = np.array(color_rgb, dtype=np.float32)
        alpha = alpha_by_class[class_name]
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    return overlay


def _select_talc_segments_with_hdbscan(
    rgb: np.ndarray,
    class_map: np.ndarray,
    interior_mask: np.ndarray,
    boundary_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    segments = _talc_segments(rgb, class_map, interior_mask, boundary_mask)
    empty_details = {
        "talcSegments": len(segments),
        "requiredSegments": 0,
        "centroidInsideSegments": 0,
        "intersectingSegments": 0,
        "selectedClusters": 0,
        "selectedSegments": 0,
        "fallback": False,
        "testedParameterSets": 0,
        "validParameterSets": 0,
        "minClusterSize": None,
        "minSamples": None,
        "featureSet": None,
        "supplementalSegments": 0,
        "insideRegionAreaMinimum": 0,
        "outsideAreaFilterThreshold": 0,
        "outsideAreaFilteredSegments": 0,
        "outsideAreaFilteredPixels": 0,
    }
    if not segments:
        return {
            "mask": np.zeros(class_map.shape, dtype=bool),
            "details": empty_details,
        }

    required_ids = {segment["id"] for segment in segments if segment["inside"]}
    intersecting_ids = {segment["id"] for segment in segments if segment["intersectsInterior"]}
    base_details = {
        **empty_details,
        "talcSegments": len(segments),
        "requiredSegments": len(required_ids),
        "centroidInsideSegments": len(required_ids),
        "intersectingSegments": len(intersecting_ids),
    }
    if not required_ids:
        return {
            "mask": np.zeros(class_map.shape, dtype=bool),
            "details": base_details,
        }

    best: dict[str, Any] | None = None
    tested_parameter_sets = 0
    valid_parameter_sets = 0
    if len(segments) >= 2:
        for feature_name, features in _standardized_segment_feature_variants(segments, rgb.shape[1], rgb.shape[0]):
            for min_cluster_size in _hdbscan_min_cluster_sizes(len(segments)):
                for min_samples in _hdbscan_min_samples(min_cluster_size):
                    tested_parameter_sets += 1
                    labels = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        allow_single_cluster=True,
                        copy=False,
                    ).fit_predict(features)
                    candidate = _hdbscan_selection_candidate(segments, labels, interior_mask, required_ids)
                    if not candidate:
                        continue
                    valid_parameter_sets += 1
                    if (
                        best is None
                        or candidate["supplemental_count"] < best["supplemental_count"]
                        or (
                            candidate["supplemental_count"] == best["supplemental_count"]
                            and candidate["outside_selected"] < best["outside_selected"]
                        )
                        or (
                            candidate["supplemental_count"] == best["supplemental_count"]
                            and candidate["outside_selected"] == best["outside_selected"]
                            and candidate["selected_count"] < best["selected_count"]
                        )
                        or (
                            candidate["supplemental_count"] == best["supplemental_count"]
                            and candidate["outside_selected"] == best["outside_selected"]
                            and candidate["selected_count"] == best["selected_count"]
                            and candidate["selected_cluster_count"] > best["selected_cluster_count"]
                        )
                    ):
                        candidate["min_cluster_size"] = min_cluster_size
                        candidate["min_samples"] = min_samples
                        candidate["feature_name"] = feature_name
                        best = candidate
                        if (
                            candidate["supplemental_count"] == 0
                            and candidate["outside_selected"] == 0
                            and candidate["selected_count"] == len(required_ids)
                        ):
                            break
                if best and best["supplemental_count"] == 0 and best["outside_selected"] == 0 and best["selected_count"] == len(required_ids):
                    break
            if best and best["supplemental_count"] == 0 and best["outside_selected"] == 0 and best["selected_count"] == len(required_ids):
                break

    if best is None:
        selected_ids = required_ids
        selected_clusters: set[int] = set()
        fallback = True
        min_cluster_size = None
        min_samples = None
        feature_name = None
        supplemental_count = 0
        selected_cluster_count = 0
    else:
        selected_ids = best["selected_ids"]
        selected_clusters = best["selected_clusters"]
        selected_cluster_count = best["selected_cluster_count"]
        fallback = False
        min_cluster_size = best["min_cluster_size"]
        min_samples = best["min_samples"]
        feature_name = best["feature_name"]
        supplemental_count = best["supplemental_count"]

    area_filter = _filter_large_outside_segments_by_region_min_area(segments, selected_ids)
    selected_ids = area_filter["selected_ids"]

    selected_mask = np.zeros(class_map.shape, dtype=bool)
    for segment in segments:
        if segment["id"] in selected_ids:
            selected_mask[segment["mask"]] = True

    details = {
        **base_details,
        "selectedClusters": selected_cluster_count,
        "selectedSegments": len(selected_ids),
        "fallback": fallback,
        "testedParameterSets": tested_parameter_sets,
        "validParameterSets": valid_parameter_sets,
        "minClusterSize": min_cluster_size,
        "minSamples": min_samples,
        "featureSet": feature_name,
        "supplementalSegments": supplemental_count,
        "insideRegionAreaMinimum": area_filter["inside_region_area_minimum"],
        "outsideAreaFilterThreshold": area_filter["threshold"],
        "outsideAreaFilteredSegments": area_filter["filtered_segments"],
        "outsideAreaFilteredPixels": area_filter["filtered_pixels"],
    }
    log = logger.warning if fallback else logger.info
    log(
        "corrected talc HDBSCAN: talc=%s required=%s intersecting=%s tested=%s valid=%s feature=%s selected_clusters=%s selected_segments=%s supplemental=%s fallback=%s elapsed=%.3fs",
        details["talcSegments"],
        details["requiredSegments"],
        details["intersectingSegments"],
        details["testedParameterSets"],
        details["validParameterSets"],
        details["featureSet"],
        details["selectedClusters"],
        details["selectedSegments"],
        details["supplementalSegments"],
        details["fallback"],
        time.perf_counter() - started_at,
    )
    return {
        "mask": selected_mask,
        "details": details,
    }


def _filter_large_outside_segments_by_region_min_area(
    segments: list[dict[str, Any]],
    selected_ids: set[int],
) -> dict[str, Any]:
    selected_id_set = set(selected_ids)
    inside_area_by_region: dict[int, int] = {}
    for segment in segments:
        if segment["id"] not in selected_id_set:
            continue
        region_id = int(segment.get("regionId", 0))
        if region_id <= 0:
            continue
        inside_area_by_region[region_id] = inside_area_by_region.get(region_id, 0) + int(segment["area"])

    if not inside_area_by_region:
        return {
            "selected_ids": selected_id_set,
            "inside_region_area_minimum": 0,
            "threshold": 0,
            "filtered_segments": 0,
            "filtered_pixels": 0,
        }

    threshold = min(inside_area_by_region.values())
    filtered_ids: set[int] = set()
    filtered_pixels = 0
    for segment in segments:
        if segment["id"] not in selected_id_set:
            continue
        if int(segment.get("regionId", 0)) != 0:
            continue
        area = int(segment["area"])
        if area > threshold:
            filtered_ids.add(int(segment["id"]))
            filtered_pixels += area

    return {
        "selected_ids": selected_id_set - filtered_ids,
        "inside_region_area_minimum": int(threshold),
        "threshold": int(threshold),
        "filtered_segments": len(filtered_ids),
        "filtered_pixels": int(filtered_pixels),
    }


def _talc_segments(
    rgb: np.ndarray,
    class_map: np.ndarray,
    interior_mask: np.ndarray,
    boundary_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    talc_mask = class_map == "talc"
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(talc_mask.astype(np.uint8), 8)
    if component_count <= 1:
        return []

    lab = color.rgb2lab(rgb / 255.0).astype(np.float32)
    segments: list[dict[str, Any]] = []
    height, width = talc_mask.shape
    _region_count, region_labels = cv2.connectedComponents(interior_mask.astype(np.uint8), 8)
    split_labels = _split_region_labels(region_labels, talc_mask, boundary_mask)
    next_segment_id = 1
    for component_id in range(1, component_count):
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        if component_area <= 0:
            continue
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        right = left + box_width
        bottom = top + box_height
        local_component = labels[top:bottom, left:right] == component_id
        local_regions = split_labels[top:bottom, left:right]

        for region_id in sorted(int(value) for value in np.unique(local_regions[local_component])):
            local_part = local_component & (local_regions == region_id)
            part_count, part_labels, part_stats, part_centroids = cv2.connectedComponentsWithStats(
                local_part.astype(np.uint8),
                8,
            )
            for part_id in range(1, part_count):
                area = int(part_stats[part_id, cv2.CC_STAT_AREA])
                if area <= 0:
                    continue
                local_mask = part_labels == part_id
                mask = np.zeros(talc_mask.shape, dtype=bool)
                mask[top:bottom, left:right] = local_mask
                interior_area = area if region_id > 0 else 0
                centroid_x = float(part_centroids[part_id][0] + left)
                centroid_y = float(part_centroids[part_id][1] + top)
                segments.append(
                    {
                        "id": next_segment_id,
                        "sourceComponentId": component_id,
                        "regionId": region_id,
                        "area": area,
                        "centroid": np.array([centroid_x, centroid_y], dtype=np.float32),
                        "mean_lab": np.mean(lab[mask], axis=0),
                        "mean_rgb": np.mean(rgb[mask], axis=0),
                        "inside": region_id > 0,
                        "intersectsInterior": interior_area > 0,
                        "interiorArea": interior_area,
                        "mask": mask,
                    }
                )
                next_segment_id += 1
    return segments


def _split_region_labels(
    region_labels: np.ndarray,
    talc_mask: np.ndarray,
    boundary_mask: np.ndarray | None,
) -> np.ndarray:
    if boundary_mask is None:
        return region_labels

    split_labels = np.array(region_labels, copy=True)
    assignable = talc_mask & boundary_mask & (split_labels == 0)
    if not np.any(assignable):
        return split_labels

    kernel = np.ones((3, 3), dtype=np.uint8)
    while True:
        previous = np.array(split_labels, copy=True)
        for region_id in sorted(int(value) for value in np.unique(previous) if int(value) > 0):
            candidates = assignable & (split_labels == 0)
            if not np.any(candidates):
                break
            expanded = cv2.dilate((previous == region_id).astype(np.uint8), kernel, iterations=1).astype(bool)
            split_labels[candidates & expanded] = region_id
        if np.array_equal(split_labels, previous):
            break
    return split_labels


def _hdbscan_min_cluster_sizes(segment_count: int) -> list[int]:
    candidates = [2, 3, 4, 5, 8, 13, 21, 34, 55, 89]
    return [value for value in candidates if value <= segment_count]


def _hdbscan_min_samples(min_cluster_size: int) -> list[int]:
    candidates = [1, 2, max(1, min_cluster_size // 2), min_cluster_size]
    return sorted(set(value for value in candidates if 1 <= value <= min_cluster_size))


def _standardized_segment_features(segments: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    features = np.array(
        [
            [
                segment["centroid"][0] / max(1.0, float(width)),
                segment["centroid"][1] / max(1.0, float(height)),
                segment["mean_lab"][0] / 100.0,
                segment["mean_lab"][1] / 128.0,
                segment["mean_lab"][2] / 128.0,
                np.log1p(float(segment["area"])) / 12.0,
            ]
            for segment in segments
        ],
        dtype=np.float32,
    )
    std = features.std(axis=0)
    return (features - features.mean(axis=0)) / np.where(std < 1e-6, 1.0, std)


def _standardized_segment_feature_variants(
    segments: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[tuple[str, np.ndarray]]:
    features = _standardized_segment_features(segments, width, height)
    variants: list[tuple[str, np.ndarray]] = [
        ("color_spatial", features),
        ("spatial", features[:, :2]),
    ]
    for spatial_weight in (2.0, 4.0, 8.0, 16.0):
        weights = np.array(
            [spatial_weight, spatial_weight, 0.5, 0.5, 0.5, 0.25],
            dtype=np.float32,
        )
        variants.append((f"spatial_x{spatial_weight:g}", features * weights))
    return variants


def _hdbscan_selection_candidate(
    segments: list[dict[str, Any]],
    labels: np.ndarray,
    interior_mask: np.ndarray,
    required_ids: set[int],
) -> dict[str, Any] | None:
    selected_clusters: set[int] = set()
    for label in sorted({int(label) for label in labels if int(label) >= 0}):
        cluster_segments = [segment for segment, segment_label in zip(segments, labels) if int(segment_label) == label]
        if not cluster_segments:
            continue
        center = _weighted_cluster_center(cluster_segments)
        if _point_in_mask(center, interior_mask):
            selected_clusters.add(label)
    if not selected_clusters:
        return None

    selected_ids = {
        segment["id"]
        for segment, label in zip(segments, labels)
        if int(label) in selected_clusters
    }
    supplemental_ids = required_ids - selected_ids
    selected_ids = selected_ids | supplemental_ids

    selected_cluster_count = len(selected_clusters) + len(supplemental_ids)
    return {
        "selected_clusters": selected_clusters,
        "selected_ids": selected_ids,
        "cluster_count": len({int(label) for label in labels if int(label) >= 0}),
        "selected_cluster_count": selected_cluster_count,
        "selected_count": len(selected_ids),
        "supplemental_count": len(supplemental_ids),
        "outside_selected": sum(
            1
            for segment in segments
            if segment["id"] in selected_ids and not segment["intersectsInterior"]
        ),
    }


def _weighted_cluster_center(segments: list[dict[str, Any]]) -> np.ndarray:
    weights = np.array([float(segment["area"]) for segment in segments], dtype=np.float32)
    points = np.array([segment["centroid"] for segment in segments], dtype=np.float32)
    return np.average(points, axis=0, weights=weights)


def _point_in_mask(point: np.ndarray, mask: np.ndarray) -> bool:
    x = int(np.clip(round(float(point[0])), 0, mask.shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, mask.shape[0] - 1))
    return bool(mask[y, x])


def _enhanced_correction_crop(
    base_box: tuple[int, int, int, int],
    interior_mask: np.ndarray,
    talc_mask: np.ndarray,
    excluded_talc_mask: np.ndarray,
    settings: CorrectionSettings,
) -> dict[str, Any]:
    left, top, right, bottom = base_box
    base_mask = np.zeros(interior_mask.shape, dtype=bool)
    base_mask[top:bottom, left:right] = True
    base_area = max(1, int(np.count_nonzero(base_mask)))
    excluded_before = int(np.count_nonzero(excluded_talc_mask & base_mask))
    mandatory_mask = _correction_crop_mandatory_mask(interior_mask, settings.crop_sketch_erosion_radius)
    mandatory_mask &= base_mask
    if not np.any(mandatory_mask):
        mandatory_mask = base_mask

    best_valid: dict[str, Any] | None = None
    best_reducing: dict[str, Any] | None = None
    for angle in _correction_crop_angles(settings.crop_max_rotation_degrees):
        for expansion in np.linspace(0.0, 1.0, 21):
            candidate = _correction_crop_candidate(
                base_box,
                base_area,
                mandatory_mask,
                talc_mask,
                excluded_talc_mask,
                excluded_before,
                float(angle),
                float(expansion),
            )
            if candidate is None or candidate["area_ratio"] < settings.crop_min_area_ratio:
                continue
            if candidate["excluded_fraction"] <= settings.crop_excluded_talc_max_fraction:
                if _better_crop_candidate(candidate, best_valid):
                    best_valid = candidate
            elif excluded_before > 0 and candidate["excluded_after"] < excluded_before:
                if _better_fallback_crop_candidate(candidate, best_reducing):
                    best_reducing = candidate

    selected = best_valid or best_reducing
    crop_fallback = best_valid is None and best_reducing is not None
    if selected is None:
        selected = {
            "mask": base_mask,
            "angle": 0.0,
            "area_ratio": 1.0,
            "excluded_after": excluded_before,
            "talc_inside_crop": int(np.count_nonzero(talc_mask & base_mask)),
            "excluded_fraction": _excluded_talc_share(excluded_before, int(np.count_nonzero(talc_mask & base_mask))),
        }
        crop_fallback = excluded_before > 0
    angle = float(selected["angle"])
    crop_type = "rotated" if abs(angle) > 1e-6 else "axis-aligned"
    enhanced = selected["excluded_after"] < excluded_before or selected["area_ratio"] < 0.999
    return {
        "mask": selected["mask"],
        "details": {
            "cropEnhanced": bool(enhanced),
            "cropFallback": bool(crop_fallback),
            "cropType": crop_type,
            "cropAngleDegrees": round(angle, 3),
            "cropAreaRatio": round(float(selected["area_ratio"]), 4),
            "cropSketchErosionRadius": int(settings.crop_sketch_erosion_radius),
            "excludedTalcPixelsBefore": excluded_before,
            "excludedTalcPixelsAfter": int(selected["excluded_after"]),
            "talcPixelsInsideCrop": int(selected["talc_inside_crop"]),
            "visibleExcludedTalcFraction": round(float(selected["excluded_fraction"]), 4),
        },
    }


def _correction_crop_mandatory_mask(interior_mask: np.ndarray, erosion_radius: int) -> np.ndarray:
    if erosion_radius <= 0:
        return interior_mask
    eroded = ndimage.distance_transform_edt(interior_mask) > float(erosion_radius)
    return eroded if np.any(eroded) else interior_mask


def max_sketch_erosion_radius(interior_mask: np.ndarray) -> int:
    if not np.any(interior_mask):
        return 0
    max_distance = float(ndimage.distance_transform_edt(interior_mask).max())
    return max(0, int(ceil(max_distance) - 1))


def _sketch_erosion_nonempty(interior_mask: np.ndarray, radius: int) -> bool:
    if radius <= 0:
        return bool(np.any(interior_mask))
    return bool(np.any(ndimage.distance_transform_edt(interior_mask) > float(radius)))


def _correction_crop_angles(max_rotation_degrees: int) -> list[float]:
    max_rotation = max(0, int(max_rotation_degrees))
    angles = [0]
    for value in range(3, max_rotation + 1, 3):
        angles.extend([-value, value])
    return [float(angle) for angle in angles]


def _correction_crop_candidate(
    base_box: tuple[int, int, int, int],
    base_area: int,
    mandatory_mask: np.ndarray,
    talc_mask: np.ndarray,
    excluded_talc_mask: np.ndarray,
    excluded_before: int,
    angle_degrees: float,
    expansion: float,
) -> dict[str, Any] | None:
    left, top, right, bottom = base_box
    points_y, points_x = np.where(mandatory_mask)
    if len(points_x) == 0:
        return None
    center = np.array([(left + right - 1) / 2.0, (top + bottom - 1) / 2.0], dtype=np.float32)
    point_coords = np.column_stack([points_x.astype(np.float32), points_y.astype(np.float32)])
    point_local = _rotate_points(point_coords, center, angle_degrees)
    base_corners = np.array(
        [[left, top], [right - 1, top], [right - 1, bottom - 1], [left, bottom - 1]],
        dtype=np.float32,
    )
    base_local = _rotate_points(base_corners, center, angle_degrees)

    tight_min = point_local.min(axis=0)
    tight_max = point_local.max(axis=0)
    base_min = base_local.min(axis=0)
    base_max = base_local.max(axis=0)
    local_min = tight_min - expansion * (tight_min - base_min)
    local_max = tight_max + expansion * (base_max - tight_max)
    local_corners = np.array(
        [
            [local_min[0], local_min[1]],
            [local_max[0], local_min[1]],
            [local_max[0], local_max[1]],
            [local_min[0], local_max[1]],
        ],
        dtype=np.float32,
    )
    corners = _inverse_rotate_points(local_corners, center, angle_degrees)
    if not _polygon_inside_box(corners, base_box):
        return None

    mask = np.zeros(mandatory_mask.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1)
    candidate_mask = mask.astype(bool)
    if np.any(mandatory_mask & ~candidate_mask):
        return None
    area = int(np.count_nonzero(candidate_mask))
    if area <= 0:
        return None
    excluded_after = int(np.count_nonzero(excluded_talc_mask & candidate_mask))
    talc_inside_crop = int(np.count_nonzero(talc_mask & candidate_mask))
    return {
        "mask": candidate_mask,
        "angle": angle_degrees,
        "area_ratio": float(area) / float(base_area),
        "excluded_after": excluded_after,
        "talc_inside_crop": talc_inside_crop,
        "excluded_fraction": _excluded_talc_share(excluded_after, talc_inside_crop),
    }


def _excluded_talc_share(excluded_talc_pixels: int, talc_pixels: int) -> float:
    return float(excluded_talc_pixels) / float(talc_pixels) if talc_pixels > 0 else 0.0


def _rotate_points(points: np.ndarray, center: np.ndarray, angle_degrees: float) -> np.ndarray:
    theta = radians(angle_degrees)
    c = cos(theta)
    s = sin(theta)
    shifted = points - center
    return np.column_stack([shifted[:, 0] * c + shifted[:, 1] * s, -shifted[:, 0] * s + shifted[:, 1] * c])


def _inverse_rotate_points(points: np.ndarray, center: np.ndarray, angle_degrees: float) -> np.ndarray:
    theta = radians(angle_degrees)
    c = cos(theta)
    s = sin(theta)
    return np.column_stack([points[:, 0] * c - points[:, 1] * s, points[:, 0] * s + points[:, 1] * c]) + center


def _polygon_inside_box(points: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = box
    return bool(
        np.all(points[:, 0] >= left - 1e-3)
        and np.all(points[:, 0] <= right - 1 + 1e-3)
        and np.all(points[:, 1] >= top - 1e-3)
        and np.all(points[:, 1] <= bottom - 1 + 1e-3)
    )


def _better_crop_candidate(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    return (
        candidate["area_ratio"] > current["area_ratio"] + 1e-6
        or (
            abs(candidate["area_ratio"] - current["area_ratio"]) <= 1e-6
            and candidate["excluded_fraction"] < current["excluded_fraction"] - 1e-6
        )
        or (
            abs(candidate["area_ratio"] - current["area_ratio"]) <= 1e-6
            and abs(candidate["excluded_fraction"] - current["excluded_fraction"]) <= 1e-6
            and abs(candidate["angle"]) < abs(current["angle"])
        )
    )


def _better_fallback_crop_candidate(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    return (
        candidate["excluded_fraction"] < current["excluded_fraction"] - 1e-6
        or (
            abs(candidate["excluded_fraction"] - current["excluded_fraction"]) <= 1e-6
            and candidate["area_ratio"] > current["area_ratio"] + 1e-6
        )
        or (
            abs(candidate["excluded_fraction"] - current["excluded_fraction"]) <= 1e-6
            and abs(candidate["area_ratio"] - current["area_ratio"]) <= 1e-6
            and abs(candidate["angle"]) < abs(current["angle"])
        )
    )


def _stats(class_map: np.ndarray) -> dict[str, float]:
    total = float(class_map.size)
    return {
        class_name: round(float(np.count_nonzero(class_map == class_name)) / total * 100.0, 2)
        for class_name in CLASS_NAMES
    }


def _masked_stats(class_map: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    total = float(np.count_nonzero(mask))
    if total <= 0:
        return {class_name: 0.0 for class_name in CLASS_NAMES}
    return {
        class_name: round(float(np.count_nonzero((class_map == class_name) & mask)) / total * 100.0, 2)
        for class_name in CLASS_NAMES
    }


def image_to_png_bytes(image: Image.Image, max_side: int = 1600) -> bytes:
    rgb = _prepare_rgb(image, max_side)
    bio = BytesIO()
    Image.fromarray(rgb).save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def extract_blue_contours_png(
    source: Image.Image,
    sketch: Image.Image,
    settings: SketchSettings | None = None,
    max_side: int = 1600,
) -> bytes:
    settings = settings or sketch_defaults()
    src = _prepare_rgb(source, max_side).astype(np.float32)
    ann = _prepare_rgb(sketch, max_side)
    overlay = src.copy()
    blue_mask = _blue_line_mask(ann)
    if np.any(blue_mask):
        marker_radius = max(4, round(max(ann.shape[:2]) / 260))
        overlay = _render_interpreted_sketch(
            overlay,
            blue_mask,
            marker_radius,
            settings.endpoint_min_branch_length,
        )
    bio = BytesIO()
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def build_sketch_geometry(
    source: Image.Image,
    sketch: Image.Image,
    settings: SketchSettings | None = None,
    saved_segments: list[dict[str, Any]] | None = None,
    region_mode: str = "inside",
) -> dict[str, Any]:
    settings = settings or sketch_defaults()
    saved_segments = saved_segments or []
    if region_mode not in {"inside", "outside"}:
        region_mode = "inside"
    width, height = source.size
    ann = np.asarray(sketch.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)
    blue_mask = _blue_line_mask(ann)
    marker_radius = max(4, round(max(width, height) / 260))
    min_branch_length = max(settings.endpoint_min_branch_length, marker_radius * 2)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(blue_mask.astype(np.uint8), 8)
    lines: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    saved_endpoint_ids = {
        str(endpoint_id)
        for segment in saved_segments
        for endpoint_id in (segment.get("aEndpointId"), segment.get("bEndpointId"))
        if endpoint_id
    }
    protected_points = {_endpoint_point_from_id(endpoint_id) for endpoint_id in saved_endpoint_ids}
    protected_points.discard(None)

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 40:
            continue

        component = labels == component_id
        skeleton = morphology.skeletonize(component)
        visible_skeleton = _visible_skeleton_without_short_spurs(skeleton, min_branch_length, protected_points)
        render_component = _render_mask_from_visible_skeleton(component, visible_skeleton)
        component_endpoint_ids: list[str] = []

        for y, x in _component_endpoints(
            visible_skeleton,
            min_branch_length=min_branch_length,
            protected_points=protected_points,
        ):
            endpoint_id = _endpoint_id(float(x), float(y))
            component_endpoint_ids.append(endpoint_id)
            endpoints.append(
                {
                    "id": endpoint_id,
                    "componentId": f"c{component_id}",
                    "x": float(x),
                    "y": float(y),
                }
            )

        lines.append(
            {
                "id": f"c{component_id}",
                "paths": _skeleton_svg_paths(visible_skeleton),
                "strokeWidth": max(2.0, round(_component_line_width(render_component), 2)),
                "endpointIds": component_endpoint_ids,
            }
        )

    endpoint_by_id = {endpoint["id"]: endpoint for endpoint in endpoints}
    normalized_segments = _normalize_saved_segments(saved_segments, endpoint_by_id, width, height)
    connected_endpoint_ids = {
        endpoint_id
        for segment in normalized_segments
        for endpoint_id in (segment.get("aEndpointId"), segment.get("bEndpointId"))
        if endpoint_id
    }
    endpoint_group_closed = _effective_endpoint_group_closed(endpoints, lines, normalized_segments, connected_endpoint_ids)

    for line in lines:
        endpoint_ids = line["endpointIds"]
        line["closed"] = len(endpoint_ids) == 0 or endpoint_group_closed.get(endpoint_ids[0], False)
        line["maskClosed"] = line["closed"]
        line["color"] = "red" if line["closed"] else "blue"

    for segment in normalized_segments:
        segment["mainColor"] = "red" if endpoint_group_closed.get(segment["aEndpointId"], False) else "blue"

    active_endpoints = [endpoint for endpoint in endpoints if endpoint["id"] not in connected_endpoint_ids]
    candidates = _connection_candidates(active_endpoints, normalized_segments, settings, width, height)
    geometry = {
        "width": width,
        "height": height,
        "lines": lines,
        "endpoints": active_endpoints,
        "allEndpoints": endpoints,
        "segments": normalized_segments,
        "candidates": candidates,
        "regionMode": region_mode,
        "settings": asdict(settings),
    }
    if _geometry_allows_region_choice(geometry):
        for line in geometry["lines"]:
            line["closed"] = True
            line["color"] = "red"
        for segment in geometry["segments"]:
            segment["mainColor"] = "red"
    geometry["regionChoices"] = _single_region_choices(geometry)
    return geometry


def render_edited_sketch_png(
    source: Image.Image,
    sketch: Image.Image,
    settings: SketchSettings | None = None,
    saved_segments: list[dict[str, Any]] | None = None,
) -> bytes:
    geometry = build_sketch_geometry(source, sketch, settings, saved_segments)
    canvas = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    blue = (0, 55, 255)

    for line in geometry["lines"]:
        stroke_width = max(1, int(round(float(line.get("strokeWidth") or max(3, geometry["width"] / 600)))))
        for path_data in line["paths"]:
            _draw_polyline(canvas, _svg_path_points(str(path_data)), blue, stroke_width)

    segment_width = _segment_stroke_width(geometry)
    for segment in geometry["segments"]:
        _draw_polyline(canvas, segment["points"], blue, segment_width)

    bio = BytesIO()
    Image.fromarray(canvas).save(bio, format="PNG", optimize=True)
    return bio.getvalue()


def _segment_stroke_width(geometry: dict[str, Any]) -> int:
    if geometry["lines"]:
        average = sum(float(line.get("strokeWidth") or 0.0) for line in geometry["lines"]) / len(geometry["lines"])
        return max(1, int(round(average)))
    return max(1, int(round(max(3, geometry["width"] / 600))))


def _svg_path_points(path_data: str) -> list[list[float]]:
    tokens = path_data.replace("M", " M ").replace("L", " L ").split()
    points: list[list[float]] = []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        if command not in {"M", "L"} or index + 2 >= len(tokens):
            index += 1
            continue
        try:
            x = float(tokens[index + 1])
            y = float(tokens[index + 2])
        except ValueError:
            index += 1
            continue
        points.append([x, y])
        index += 3
    return points


def _draw_polyline(canvas: np.ndarray, points: list[list[float]], color_rgb: tuple[int, int, int], stroke_width: int) -> None:
    if len(points) < 2:
        return
    coords = np.array([[round(float(point[0])), round(float(point[1]))] for point in points], dtype=np.int32)
    cv2.polylines(
        canvas,
        [coords],
        isClosed=False,
        color=color_rgb,
        thickness=stroke_width,
        lineType=cv2.LINE_AA,
    )


def _closed_geometry_region(geometry: dict[str, Any]) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    masks = _closed_geometry_masks(geometry)
    if masks is None:
        return None
    crop_box, interior, _boundary_mask = masks
    return crop_box, interior


def _closed_geometry_masks(geometry: dict[str, Any]) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    lines = geometry["lines"]
    if not lines:
        return None

    width = int(geometry["width"])
    height = int(geometry["height"])
    line_mask = _closed_geometry_line_mask(geometry)
    base_interior = _closed_geometry_base_interior(line_mask)
    if base_interior is None:
        return None
    base_region_count = _single_region_count(base_interior)
    has_open_geometry = bool(geometry["endpoints"] or any(not line.get("closed") for line in lines))
    allows_choice = _geometry_allows_region_choice(geometry, base_interior)
    if has_open_geometry and (base_region_count != 1 or not _open_edge_geometry_allows_region_choice(geometry)):
        return None
    interior = base_interior
    if geometry.get("regionMode") == "outside" and allows_choice:
        interior = ~(base_interior | (line_mask > 0))
    if not np.any(interior):
        return None
    ys, xs = np.where(interior | (line_mask > 0))
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), interior, line_mask > 0


def _closed_geometry_line_mask(geometry: dict[str, Any]) -> np.ndarray:
    width = int(geometry["width"])
    height = int(geometry["height"])
    line_mask = np.zeros((height, width), dtype=np.uint8)
    for line in geometry["lines"]:
        stroke_width = max(3, int(round(float(line.get("strokeWidth") or max(3, width / 600)))))
        for path_data in line["paths"]:
            _draw_mask_polyline(line_mask, _svg_path_points(str(path_data)), stroke_width, False)

    segment_width = _segment_stroke_width(geometry)
    for segment in geometry["segments"]:
        _draw_mask_polyline(line_mask, segment["points"], segment_width, False)
    return line_mask


def _closed_geometry_base_interior(line_mask: np.ndarray) -> np.ndarray | None:
    height, width = line_mask.shape
    free_space = np.where(line_mask > 0, 0, 255).astype(np.uint8)
    exterior = free_space.copy()
    fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    for x in range(width):
        if exterior[0, x] == 255:
            cv2.floodFill(exterior, fill_mask, (x, 0), 128)
        if exterior[height - 1, x] == 255:
            cv2.floodFill(exterior, fill_mask, (x, height - 1), 128)
    for y in range(height):
        if exterior[y, 0] == 255:
            cv2.floodFill(exterior, fill_mask, (0, y), 128)
        if exterior[y, width - 1] == 255:
            cv2.floodFill(exterior, fill_mask, (width - 1, y), 128)

    interior = exterior == 255
    if not np.any(interior):
        return None
    return interior


def _single_region_choices(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    lines = geometry["lines"]
    if not lines:
        return []
    if geometry.get("endpoints") and not _open_edge_geometry_allows_region_choice(geometry):
        return []
    line_mask = _closed_geometry_line_mask(geometry)
    interior = _closed_geometry_base_interior(line_mask)
    if interior is None or not _geometry_allows_region_choice(geometry, interior):
        return []
    inside_path = _mask_svg_path(interior)
    if not inside_path:
        return []
    width = int(geometry["width"])
    height = int(geometry["height"])
    outer_path = f"M 0 0 L {width} 0 L {width} {height} L 0 {height} L 0 0"
    selected = geometry.get("regionMode") if geometry.get("regionMode") in {"inside", "outside"} else "inside"
    return [
        {"id": "inside", "label": "Внутренняя область", "path": inside_path, "fillRule": "evenodd", "selected": selected == "inside"},
        {
            "id": "outside",
            "label": "Все кроме внутренней области",
            "path": f"{outer_path} {inside_path}",
            "fillRule": "evenodd",
            "selected": selected == "outside",
        },
    ]


def _single_region_count(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return max(0, int(count) - 1)


def _mask_svg_path(mask: np.ndarray) -> str:
    contours, _hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parts: list[str] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        start = points[0]
        commands = [f"M {int(start[0])} {int(start[1])}"]
        commands.extend(f"L {int(point[0])} {int(point[1])}" for point in points[1:])
        commands.append("Z")
        parts.append(" ".join(commands))
    return " ".join(parts)


def _geometry_allows_region_choice(geometry: dict[str, Any], interior: np.ndarray | None = None) -> bool:
    if not geometry["lines"]:
        return False
    if geometry.get("endpoints") and not _open_edge_geometry_allows_region_choice(geometry):
        return False
    if interior is None:
        line_mask = _closed_geometry_line_mask(geometry)
        interior = _closed_geometry_base_interior(line_mask)
    if interior is None:
        return False
    if _single_region_count(interior) == 1:
        return True
    return _geometry_has_single_closed_line_component(geometry)


def _geometry_has_single_closed_line_component(geometry: dict[str, Any]) -> bool:
    lines = geometry.get("lines") or []
    return (
        len(lines) == 1
        and not geometry.get("endpoints")
        and bool(lines[0].get("closed"))
    )


def _open_edge_geometry_allows_region_choice(geometry: dict[str, Any]) -> bool:
    if not geometry.get("endpoints"):
        return True
    return len(geometry.get("lines") or []) == 1 and _active_endpoints_are_on_edge(geometry)


def _active_endpoints_are_on_edge(geometry: dict[str, Any]) -> bool:
    endpoints = geometry.get("endpoints") or []
    if not endpoints:
        return True
    width = float(geometry["width"])
    height = float(geometry["height"])
    tolerance = max(2.0, min(width, height) * 0.01)
    for endpoint in endpoints:
        x = float(endpoint.get("x", 0.0))
        y = float(endpoint.get("y", 0.0))
        if min(abs(x), abs(x - width), abs(y), abs(y - height)) > tolerance:
            return False
    return True


def _draw_mask_polyline(mask: np.ndarray, points: list[list[float]], stroke_width: int, closed: bool) -> None:
    if len(points) < 2:
        return
    coords = np.array([[round(float(point[0])), round(float(point[1]))] for point in points], dtype=np.int32)
    cv2.polylines(mask, [coords], isClosed=closed, color=255, thickness=stroke_width, lineType=cv2.LINE_AA)


def _scale_box(
    box: tuple[int, int, int, int],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    x_scale = target_width / float(source_width)
    y_scale = target_height / float(source_height)
    scaled_left = max(0, min(target_width - 1, int(np.floor(left * x_scale))))
    scaled_top = max(0, min(target_height - 1, int(np.floor(top * y_scale))))
    scaled_right = max(scaled_left + 1, min(target_width, int(np.ceil(right * x_scale))))
    scaled_bottom = max(scaled_top + 1, min(target_height, int(np.ceil(bottom * y_scale))))
    return scaled_left, scaled_top, scaled_right, scaled_bottom


def _resize_bool_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def _blue_line_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 45)
    close_kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(blue.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close_kernel) > 0


def _skeleton_svg_paths(skeleton: np.ndarray) -> list[str]:
    points = {(int(y), int(x)) for y, x in np.argwhere(skeleton)}
    if not points:
        return []

    neighbors = {point: [neighbor for neighbor in _skeleton_neighbors(skeleton, *point)] for point in points}
    nodes = {point for point, point_neighbors in neighbors.items() if len(point_neighbors) != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[str] = []

    if nodes:
        for node in sorted(nodes):
            for neighbor in neighbors[node]:
                edge = _edge_key(node, neighbor)
                if edge in visited_edges:
                    continue
                path = _trace_skeleton_path(node, neighbor, neighbors, visited_edges)
                if len(path) > 1:
                    paths.append(_points_to_svg_path(path))
    else:
        start = min(points)
        neighbor = neighbors[start][0]
        path = _trace_skeleton_loop(start, neighbor, neighbors, visited_edges)
        if len(path) > 1:
            paths.append(_points_to_svg_path(path))

    return paths


def _trace_skeleton_path(
    start: tuple[int, int],
    next_point: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    path = [start]
    previous = start
    current = next_point

    while True:
        visited_edges.add(_edge_key(previous, current))
        path.append(current)
        if len(neighbors[current]) != 2:
            return path
        candidates = [neighbor for neighbor in neighbors[current] if neighbor != previous]
        if len(candidates) != 1:
            return path
        previous, current = current, candidates[0]
        if _edge_key(previous, current) in visited_edges:
            return path


def _trace_skeleton_loop(
    start: tuple[int, int],
    next_point: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    path = [start]
    previous = start
    current = next_point

    while True:
        visited_edges.add(_edge_key(previous, current))
        path.append(current)
        candidates = [neighbor for neighbor in neighbors[current] if neighbor != previous]
        if not candidates:
            return path
        next_candidate = candidates[0]
        if next_candidate == start:
            path.append(start)
            return path
        previous, current = current, next_candidate
        if _edge_key(previous, current) in visited_edges:
            return path


def _points_to_svg_path(points: list[tuple[int, int]]) -> str:
    first_y, first_x = points[0]
    parts = [f"M {first_x} {first_y}"]
    parts.extend(f"L {x} {y}" for y, x in points[1:])
    return " ".join(parts)


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def _normalize_saved_segments(
    saved_segments: list[dict[str, Any]],
    endpoint_by_id: dict[str, dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in saved_segments:
        a_endpoint_id = str(segment.get("aEndpointId", ""))
        b_endpoint_id = str(segment.get("bEndpointId", ""))
        a_endpoint = endpoint_by_id.get(a_endpoint_id)
        b_endpoint = endpoint_by_id.get(b_endpoint_id)
        if not a_endpoint or not b_endpoint:
            continue
        route_type = str(segment.get("routeType", ""))
        control = _coerce_point(segment.get("control"))
        tangent = _coerce_point(segment.get("tangent"))
        if route_type != "curve" or control is None or tangent is None:
            continue
        routes = _connection_routes(a_endpoint, b_endpoint, width, height)
        control_point = [float(control[0]), float(control[1])]
        tangent_vector = [round(float(tangent[0]), 2), round(float(tangent[1]), 2)]
        route_points = (
            routes["perimeter"]["points"]
            if _curve_uses_perimeter_route(control_point, tangent_vector, float(width), float(height))
            else _cubic_curve_points(
                routes["straight"]["points"][0],
                control_point,
                tangent_vector,
                routes["straight"]["points"][1],
                float(width),
                float(height),
            )
        )
        normalized.append(
            {
                "id": str(segment.get("id", "")),
                "a": [a_endpoint["x"], a_endpoint["y"]],
                "b": [b_endpoint["x"], b_endpoint["y"]],
                "aEndpointId": a_endpoint_id,
                "bEndpointId": b_endpoint_id,
                "routeType": "curve",
                "control": control_point,
                "tangent": tangent_vector,
                "points": route_points,
            }
        )
    return normalized


def _effective_endpoint_group_closed(
    endpoints: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    connected_endpoint_ids: set[str],
) -> dict[str, bool]:
    endpoint_ids = [str(endpoint["id"]) for endpoint in endpoints]
    if not endpoint_ids:
        return {}

    groups = _EndpointGroups(endpoint_ids)
    for line in lines:
        line_endpoint_ids = [str(endpoint_id) for endpoint_id in line["endpointIds"] if endpoint_id in groups.parent]
        for endpoint_id in line_endpoint_ids[1:]:
            groups.union(line_endpoint_ids[0], endpoint_id)

    for segment in segments:
        a_endpoint_id = str(segment.get("aEndpointId", ""))
        b_endpoint_id = str(segment.get("bEndpointId", ""))
        if a_endpoint_id in groups.parent and b_endpoint_id in groups.parent:
            groups.union(a_endpoint_id, b_endpoint_id)

    endpoint_ids_by_root: dict[str, list[str]] = {}
    for endpoint_id in endpoint_ids:
        endpoint_ids_by_root.setdefault(groups.find(endpoint_id), []).append(endpoint_id)

    root_closed = {
        root: all(endpoint_id in connected_endpoint_ids for endpoint_id in root_endpoint_ids)
        for root, root_endpoint_ids in endpoint_ids_by_root.items()
    }
    return {endpoint_id: root_closed[groups.find(endpoint_id)] for endpoint_id in endpoint_ids}


class _EndpointGroups:
    def __init__(self, endpoint_ids: list[str]) -> None:
        self.parent = {endpoint_id: endpoint_id for endpoint_id in endpoint_ids}

    def find(self, endpoint_id: str) -> str:
        parent = self.parent[endpoint_id]
        if parent != endpoint_id:
            parent = self.find(parent)
            self.parent[endpoint_id] = parent
        return parent

    def union(self, a_endpoint_id: str, b_endpoint_id: str) -> None:
        a_root = self.find(a_endpoint_id)
        b_root = self.find(b_endpoint_id)
        if a_root != b_root:
            self.parent[b_root] = a_root


def _connection_candidates(
    endpoints: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    settings: SketchSettings,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    existing_pairs = {
        tuple(sorted((segment.get("aEndpointId"), segment.get("bEndpointId"))))
        for segment in segments
        if segment.get("aEndpointId") and segment.get("bEndpointId")
    }
    candidate_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

    for endpoint in endpoints:
        ranked: list[tuple[float, dict[str, Any]]] = []
        for other in endpoints:
            if endpoint["id"] == other["id"]:
                continue
            distance = float(np.hypot(endpoint["x"] - other["x"], endpoint["y"] - other["y"]))
            if distance <= settings.connection_max_distance:
                ranked.append((distance, other))

        for distance, other in sorted(ranked, key=lambda item: item[0])[: settings.connection_neighbors_per_endpoint]:
            _add_connection_candidate(
                candidate_by_pair,
                existing_pairs,
                endpoint,
                other,
                "distance",
                width,
                height,
            )

    if settings.angular_neighbors_per_endpoint > 0 and len(endpoints) > 1:
        center_x = width / 2.0
        center_y = height / 2.0
        angular_endpoints = sorted(
            endpoints,
            key=lambda endpoint: float(np.arctan2(endpoint["y"] - center_y, endpoint["x"] - center_x)),
        )
        endpoint_count = len(angular_endpoints)
        for index, endpoint in enumerate(angular_endpoints):
            for offset in range(1, min(settings.angular_neighbors_per_endpoint, endpoint_count - 1) + 1):
                for neighbor_index in ((index - offset) % endpoint_count, (index + offset) % endpoint_count):
                    other = angular_endpoints[neighbor_index]
                    if other["id"] == endpoint["id"]:
                        continue
                    _add_connection_candidate(
                        candidate_by_pair,
                        existing_pairs,
                        endpoint,
                        other,
                        "angular",
                        width,
                        height,
                    )

    return sorted(candidate_by_pair.values(), key=lambda candidate: (candidate["distance"], candidate["id"]))


def _add_connection_candidate(
    candidate_by_pair: dict[tuple[str, str], dict[str, Any]],
    existing_pairs: set[tuple[str, str]],
    endpoint: dict[str, Any],
    other: dict[str, Any],
    reason: str,
    width: int,
    height: int,
) -> None:
    pair = tuple(sorted((endpoint["id"], other["id"])))
    if pair in existing_pairs:
        return

    distance = round(float(np.hypot(endpoint["x"] - other["x"], endpoint["y"] - other["y"])), 2)
    if pair not in candidate_by_pair:
        routes = _connection_routes(endpoint, other, width, height)
        control_bounds = _control_bounds(routes)
        straight_only = _control_bounds_degenerate(control_bounds)
        candidate_by_pair[pair] = {
            "id": f"{pair[0]}--{pair[1]}",
            "aEndpointId": endpoint["id"],
            "bEndpointId": other["id"],
            "a": [endpoint["x"], endpoint["y"]],
            "b": [other["x"], other["y"]],
            "distance": distance,
            "reason": reason,
            "straightRoute": routes["straight"],
            "perimeterRoute": routes["perimeter"],
            "controlBounds": control_bounds,
            "curveDefaultControl": _straight_default_control(routes) if straight_only else _curve_default_control(routes),
            "curveDefaultTangent": _curve_default_tangent(routes),
            "straightOnly": straight_only,
        }
        return

    current_reason = candidate_by_pair[pair]["reason"]
    if reason not in current_reason.split("+"):
        candidate_by_pair[pair]["reason"] = "+".join(sorted([*current_reason.split("+"), reason]))


def _connection_routes(
    endpoint: dict[str, Any],
    other: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, dict[str, Any]]:
    a = [float(endpoint["x"]), float(endpoint["y"])]
    b = [float(other["x"]), float(other["y"])]
    perimeter_points = _perimeter_route_points(a, b, float(width), float(height))
    return {
        "straight": {"routeType": "straight", "points": [a, b]},
        "perimeter": {"routeType": "perimeter", "points": perimeter_points},
    }


def _control_bounds(routes: dict[str, dict[str, Any]]) -> list[list[float]]:
    straight_points = routes["straight"]["points"]
    perimeter_points = routes["perimeter"]["points"]
    return _dedupe_route_points([straight_points[0], straight_points[1], *reversed(perimeter_points[1:-1])])


def _control_bounds_degenerate(bounds: list[list[float]]) -> bool:
    if len(bounds) < 3:
        return True
    return abs(_polygon_area(bounds)) <= 1e-6


def _curve_default_control(routes: dict[str, dict[str, Any]]) -> list[float]:
    straight_points = routes["straight"]["points"]
    perimeter_points = routes["perimeter"]["points"]
    straight_center = _polyline_midpoint(straight_points)
    perimeter_center = _polyline_midpoint(perimeter_points[1:-1] or perimeter_points)
    return [
        round((straight_center[0] + perimeter_center[0]) / 2.0, 2),
        round((straight_center[1] + perimeter_center[1]) / 2.0, 2),
    ]


def _straight_default_control(routes: dict[str, dict[str, Any]]) -> list[float]:
    return _polyline_midpoint(routes["straight"]["points"])


def _curve_default_tangent(routes: dict[str, dict[str, Any]]) -> list[float]:
    a, b = routes["straight"]["points"]
    return [round(float(b[0] - a[0]), 2), round(float(b[1] - a[1]), 2)]


def _cubic_curve_points(
    a: list[float],
    control: list[float],
    tangent: list[float],
    b: list[float],
    width: float,
    height: float,
    steps: int = 64,
) -> list[list[float]]:
    first_control, second_control = _cubic_controls_for_midpoint_tangent(a, control, tangent, b)
    points: list[list[float]] = []
    for index in range(steps + 1):
        t = index / steps
        mt = 1.0 - t
        x = mt**3 * a[0] + 3.0 * mt * mt * t * first_control[0] + 3.0 * mt * t * t * second_control[0] + t**3 * b[0]
        y = mt**3 * a[1] + 3.0 * mt * mt * t * first_control[1] + 3.0 * mt * t * t * second_control[1] + t**3 * b[1]
        points.append([round(float(x), 2), round(float(y), 2)])
    return _clamp_polyline_to_box(points, width, height)


def _cubic_controls_for_midpoint_tangent(
    a: list[float],
    midpoint: list[float],
    tangent: list[float],
    b: list[float],
) -> tuple[list[float], list[float]]:
    controls_sum = [
        (8.0 * midpoint[0] - a[0] - b[0]) / 3.0,
        (8.0 * midpoint[1] - a[1] - b[1]) / 3.0,
    ]
    controls_delta = [
        (4.0 / 3.0) * tangent[0] - (b[0] - a[0]),
        (4.0 / 3.0) * tangent[1] - (b[1] - a[1]),
    ]
    return [
        (controls_sum[0] - controls_delta[0]) / 2.0,
        (controls_sum[1] - controls_delta[1]) / 2.0,
    ], [
        (controls_sum[0] + controls_delta[0]) / 2.0,
        (controls_sum[1] + controls_delta[1]) / 2.0,
    ]


def _curve_uses_perimeter_route(control: list[float], tangent: list[float], width: float, height: float) -> bool:
    edge_tolerance = _image_edge_tolerance(width, height)
    if not _point_on_image_edge(control, width, height, edge_tolerance):
        return False
    handles = (
        [control[0] + tangent[0] / 4.0, control[1] + tangent[1] / 4.0],
        [control[0] - tangent[0] / 4.0, control[1] - tangent[1] / 4.0],
    )
    return any(_point_on_or_outside_image_edge(handle, width, height, edge_tolerance) for handle in handles)


def _image_edge_tolerance(width: float, height: float) -> float:
    return max(2.0, min(width, height) * 0.015)


def _point_on_image_edge(point: list[float], width: float, height: float, eps: float) -> bool:
    x, y = point
    return (
        -eps <= x <= width + eps
        and -eps <= y <= height + eps
        and (abs(x) <= eps or abs(x - width) <= eps or abs(y) <= eps or abs(y - height) <= eps)
    )


def _point_on_or_outside_image_edge(point: list[float], width: float, height: float, eps: float) -> bool:
    x, y = point
    return x <= eps or x >= width - eps or y <= eps or y >= height - eps


def sketch_control_in_bounds(control: tuple[float, float], bounds: list[list[float]]) -> bool:
    point = [float(control[0]), float(control[1])]
    if len(bounds) == 1:
        return float(np.hypot(point[0] - bounds[0][0], point[1] - bounds[0][1])) <= 1e-6
    if len(bounds) == 2:
        return _point_on_segment(point, bounds[0], bounds[1])
    return _point_in_polygon_or_on_edge(point, bounds)


def _clamp_polyline_to_box(points: list[list[float]], width: float, height: float) -> list[list[float]]:
    if not points:
        return []

    result: list[list[float]] = []
    pending_exit: list[float] | None = None
    if _point_inside_box(points[0], width, height):
        result.append(_round_point(points[0]))

    for start, end in zip(points, points[1:]):
        start_inside = _point_inside_box(start, width, height)
        end_inside = _point_inside_box(end, width, height)
        clipped = _clip_segment_to_box(start, end, width, height)

        if start_inside and end_inside:
            _append_point(result, end)
            continue

        if start_inside and not end_inside:
            if clipped:
                _t0, _t1, _entry, exit_point = clipped
                _append_point(result, exit_point)
                pending_exit = exit_point
            continue

        if not start_inside and end_inside:
            if clipped:
                _t0, _t1, entry, _exit_point = clipped
                if pending_exit is not None:
                    _append_points(result, _shortest_perimeter_arc_points(pending_exit, entry, width, height)[1:])
                    pending_exit = None
                else:
                    _append_point(result, entry)
                _append_point(result, end)
            continue

        if clipped:
            _t0, _t1, entry, exit_point = clipped
            if pending_exit is not None:
                _append_points(result, _shortest_perimeter_arc_points(pending_exit, entry, width, height)[1:])
                pending_exit = None
            _append_point(result, entry)
            _append_point(result, exit_point)
            pending_exit = exit_point

    return result


def _point_inside_box(point: list[float], width: float, height: float) -> bool:
    return -1e-9 <= point[0] <= width + 1e-9 and -1e-9 <= point[1] <= height + 1e-9


def _clip_segment_to_box(
    start: list[float],
    end: list[float],
    width: float,
    height: float,
) -> tuple[float, float, list[float], list[float]] | None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    t0 = 0.0
    t1 = 1.0
    for p, q in ((-dx, x0), (dx, width - x0), (-dy, y0), (dy, height - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            if ratio > t1:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0:
                return None
            t1 = min(t1, ratio)
    if t0 > t1:
        return None
    return (
        t0,
        t1,
        _point_at_segment_t(start, end, t0),
        _point_at_segment_t(start, end, t1),
    )


def _point_at_segment_t(start: list[float], end: list[float], t: float) -> list[float]:
    return [
        round(float(start[0] + (end[0] - start[0]) * t), 2),
        round(float(start[1] + (end[1] - start[1]) * t), 2),
    ]


def _append_points(points: list[list[float]], new_points: list[list[float]]) -> None:
    for point in new_points:
        _append_point(points, point)


def _append_point(points: list[list[float]], point: list[float]) -> None:
    rounded = _round_point(point)
    if not points or abs(points[-1][0] - rounded[0]) > 1e-6 or abs(points[-1][1] - rounded[1]) > 1e-6:
        points.append(rounded)


def _round_point(point: list[float]) -> list[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _perimeter_route_points(a: list[float], b: list[float], width: float, height: float) -> list[list[float]]:
    center = [width / 2.0, height / 2.0]
    a_border = _ray_box_intersection(center, a, width, height)
    b_border = _ray_box_intersection(center, b, width, height)
    clockwise = _perimeter_arc_points(a_border, b_border, width, height, True)
    counter = _perimeter_arc_points(a_border, b_border, width, height, False)
    arc = _choose_inner_perimeter_arc(clockwise, counter, center, a, b)
    return _dedupe_route_points([a, *arc, b])


def _ray_box_intersection(center: list[float], point: list[float], width: float, height: float) -> list[float]:
    cx, cy = center
    dx = point[0] - cx
    dy = point[1] - cy
    candidates: list[tuple[float, float, float]] = []
    if abs(dx) > 1e-9:
        for x in (0.0, width):
            t = (x - cx) / dx
            y = cy + t * dy
            if t > 0 and -1e-6 <= y <= height + 1e-6:
                candidates.append((t, x, min(height, max(0.0, y))))
    if abs(dy) > 1e-9:
        for y in (0.0, height):
            t = (y - cy) / dy
            x = cx + t * dx
            if t > 0 and -1e-6 <= x <= width + 1e-6:
                candidates.append((t, min(width, max(0.0, x)), y))
    if not candidates:
        return [point[0], point[1]]
    _t, x, y = min(candidates, key=lambda item: item[0])
    return [round(float(x), 2), round(float(y), 2)]


def _perimeter_arc_points(
    start: list[float],
    end: list[float],
    width: float,
    height: float,
    clockwise: bool,
) -> list[list[float]]:
    total = 2.0 * (width + height)
    start_pos = _perimeter_position(start, width, height)
    end_pos = _perimeter_position(end, width, height)
    if clockwise:
        distance = (end_pos - start_pos) % total
        positions = [start_pos, *_corner_positions_between(start_pos, distance, total, width, height), start_pos + distance]
    else:
        distance = (start_pos - end_pos) % total
        positions = [start_pos, *_corner_positions_between(start_pos, -distance, total, width, height), start_pos - distance]
    return [_perimeter_point(position % total, width, height) for position in positions]


def _shortest_perimeter_arc_points(start: list[float], end: list[float], width: float, height: float) -> list[list[float]]:
    clockwise = _perimeter_arc_points(start, end, width, height, True)
    counter = _perimeter_arc_points(start, end, width, height, False)
    return min((clockwise, counter), key=_polyline_length)


def _corner_positions_between(
    start_pos: float,
    signed_distance: float,
    total: float,
    width: float,
    height: float,
) -> list[float]:
    if abs(signed_distance) < 1e-9:
        return []
    direction = 1.0 if signed_distance > 0 else -1.0
    distance = abs(signed_distance)
    corners = [0.0, width, width + height, 2.0 * width + height]
    positions: list[float] = []
    for corner in corners:
        delta = ((corner - start_pos) * direction) % total
        if 1e-6 < delta < distance - 1e-6:
            positions.append(start_pos + direction * delta)
    return sorted(positions, reverse=direction < 0)


def _perimeter_position(point: list[float], width: float, height: float) -> float:
    x, y = point
    distances = [
        (abs(y), x),
        (abs(x - width), width + y),
        (abs(y - height), width + height + (width - x)),
        (abs(x), 2.0 * width + height + (height - y)),
    ]
    return float(min(distances, key=lambda item: item[0])[1])


def _perimeter_point(position: float, width: float, height: float) -> list[float]:
    position %= 2.0 * (width + height)
    if position <= width:
        return [round(position, 2), 0.0]
    position -= width
    if position <= height:
        return [round(width, 2), round(position, 2)]
    position -= height
    if position <= width:
        return [round(width - position, 2), round(height, 2)]
    position -= width
    return [0.0, round(height - position, 2)]


def _choose_inner_perimeter_arc(
    clockwise: list[list[float]],
    counter: list[list[float]],
    center: list[float],
    a: list[float],
    b: list[float],
) -> list[list[float]]:
    a_angle = _angle_from_center(center, a)
    b_angle = _angle_from_center(center, b)
    candidates = []
    for arc in (clockwise, counter):
        midpoint = _polyline_midpoint(arc)
        midpoint_angle = _angle_from_center(center, midpoint)
        candidates.append((_angle_is_between_smaller(a_angle, b_angle, midpoint_angle), _polyline_length(arc), arc))
    inside = [candidate for candidate in candidates if candidate[0]]
    return min(inside or candidates, key=lambda candidate: candidate[1])[2]


def _angle_from_center(center: list[float], point: list[float]) -> float:
    return float(np.arctan2(point[1] - center[1], point[0] - center[0]) % tau)


def _angle_is_between_smaller(a_angle: float, b_angle: float, value: float) -> bool:
    delta = (b_angle - a_angle) % tau
    if delta <= pi:
        return (value - a_angle) % tau <= delta + 1e-9
    return (a_angle - value) % tau <= (tau - delta) + 1e-9


def _polyline_midpoint(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0]
    total = _polyline_length(points)
    if total <= 1e-9:
        return [points[0][0], points[0][1]]
    target = total / 2.0
    travelled = 0.0
    for start, end in zip(points, points[1:]):
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        if travelled + length >= target and length > 1e-9:
            ratio = (target - travelled) / length
            return [round(start[0] + (end[0] - start[0]) * ratio, 2), round(start[1] + (end[1] - start[1]) * ratio, 2)]
        travelled += length
    return [points[-1][0], points[-1][1]]


def _polyline_length(points: list[list[float]]) -> float:
    return float(sum(np.hypot(end[0] - start[0], end[1] - start[1]) for start, end in zip(points, points[1:])))


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    previous = points[-1]
    for current in points:
        area += previous[0] * current[1] - current[0] * previous[1]
        previous = current
    return area / 2.0


def _point_in_polygon_or_on_edge(point: list[float], polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        xi, yi = current
        xj, yj = previous
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x <= x_intersection + 1e-9:
                inside = not inside
        previous = current
    return inside


def _point_on_segment(point: list[float], start: list[float], end: list[float]) -> bool:
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-6:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -1e-6:
        return False
    squared_length = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= squared_length + 1e-6


def _dedupe_route_points(points: list[list[float]]) -> list[list[float]]:
    deduped: list[list[float]] = []
    for point in points:
        rounded = [round(float(point[0]), 2), round(float(point[1]), 2)]
        if not deduped or abs(deduped[-1][0] - rounded[0]) > 1e-6 or abs(deduped[-1][1] - rounded[1]) > 1e-6:
            deduped.append(rounded)
    return deduped


def _coerce_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _point_key(value: float) -> int:
    return int(round(value))


def _endpoint_id(x: float, y: float) -> str:
    return f"e_{_point_key(x)}_{_point_key(y)}"


def _endpoint_point_from_id(endpoint_id: str) -> tuple[int, int] | None:
    parts = endpoint_id.split("_")
    if len(parts) != 3 or parts[0] != "e":
        return None
    try:
        x = int(parts[1])
        y = int(parts[2])
    except ValueError:
        return None
    return y, x


def _render_interpreted_sketch(
    overlay: np.ndarray,
    blue_mask: np.ndarray,
    marker_radius: int,
    endpoint_min_branch_length: int,
) -> np.ndarray:
    result = overlay.copy()
    blue_rgb = np.array((0, 55, 255), dtype=np.float32)
    red_rgb = np.array((255, 40, 40), dtype=np.float32)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(blue_mask.astype(np.uint8), 8)
    endpoint_markers = np.zeros(blue_mask.shape, dtype=np.uint8)

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 40:
            continue

        component = labels == component_id
        skeleton = morphology.skeletonize(component)
        visible_skeleton = _visible_skeleton_without_short_spurs(
            skeleton,
            min_branch_length=max(endpoint_min_branch_length, marker_radius * 2),
        )
        render_component = _render_mask_from_visible_skeleton(component, visible_skeleton)
        endpoints = _component_endpoints(
            visible_skeleton,
            min_branch_length=max(endpoint_min_branch_length, marker_radius * 2),
        )
        if endpoints:
            result[render_component] = result[render_component] * 0.15 + blue_rgb * 0.85
            for y, x in endpoints:
                cv2.circle(endpoint_markers, (int(x), int(y)), marker_radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        else:
            result[render_component] = result[render_component] * 0.12 + red_rgb * 0.88

    marker_mask = endpoint_markers > 0
    result[marker_mask] = result[marker_mask] * 0.1 + red_rgb * 0.9
    return result


def _component_endpoints(
    component_or_skeleton: np.ndarray,
    min_branch_length: int = 12,
    protected_points: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    protected_points = protected_points or set()
    skeleton = component_or_skeleton if _looks_like_skeleton(component_or_skeleton) else morphology.skeletonize(component_or_skeleton)
    if not np.any(skeleton):
        return []

    padded = np.pad(skeleton.astype(np.uint8), 1)
    endpoints: list[tuple[int, int]] = []
    for y, x in np.argwhere(skeleton):
        yy, xx = int(y) + 1, int(x) + 1
        neighbor_count = int(padded[yy - 1 : yy + 2, xx - 1 : xx + 2].sum()) - 1
        point = (int(y), int(x))
        if neighbor_count == 1 and (
            point in protected_points
            or _endpoint_branch_length(skeleton, int(y), int(x), min_branch_length) >= min_branch_length
        ):
            endpoints.append((int(y), int(x)))
    return endpoints


def _visible_skeleton_without_short_spurs(
    skeleton: np.ndarray,
    min_branch_length: int,
    protected_points: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    visible = skeleton.copy()
    if min_branch_length <= 0:
        return visible

    spur_pixels = _short_spur_pixels(skeleton, min_branch_length, protected_points or set())
    for y, x in spur_pixels:
        visible[y, x] = False
    if np.any(visible):
        return visible
    return skeleton


def _render_mask_from_visible_skeleton(component: np.ndarray, visible_skeleton: np.ndarray) -> np.ndarray:
    if not np.any(visible_skeleton):
        return component.copy()

    line_radius = max(1, int(round(_component_line_width(component) / 2)))
    visible_line = cv2.dilate(
        visible_skeleton.astype(np.uint8) * 255,
        morphology.disk(line_radius).astype(np.uint8),
        iterations=1,
    ) > 0
    return (component & visible_line) | visible_skeleton


def _short_spur_pixels(
    skeleton: np.ndarray,
    min_branch_length: int,
    protected_points: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]]:
    protected_points = protected_points or set()
    pixels: set[tuple[int, int]] = set()
    for y, x in np.argwhere(skeleton):
        if (int(y), int(x)) in protected_points:
            continue
        if len(_skeleton_neighbors(skeleton, int(y), int(x))) != 1:
            continue

        path, ended_at_junction = _endpoint_branch_path(skeleton, int(y), int(x), min_branch_length)
        if ended_at_junction and len(path) < min_branch_length:
            pixels.update(path)
    return pixels


def _endpoint_branch_path(
    skeleton: np.ndarray,
    start_y: int,
    start_x: int,
    limit: int,
) -> tuple[list[tuple[int, int]], bool]:
    previous: tuple[int, int] | None = None
    current = (start_y, start_x)
    path: list[tuple[int, int]] = []

    while len(path) < limit:
        neighbors = _skeleton_neighbors(skeleton, *current)
        if previous is not None and len(neighbors) != 2:
            return path, len(neighbors) > 2

        candidates = [neighbor for neighbor in neighbors if neighbor != previous]
        if len(candidates) != 1:
            return path, False

        path.append(current)
        previous = current
        current = candidates[0]

    return path, False


def _component_line_width(component: np.ndarray) -> float:
    skeleton = morphology.skeletonize(component)
    skeleton_pixels = int(np.count_nonzero(skeleton))
    if skeleton_pixels == 0:
        return 1.0
    return max(1.0, float(np.count_nonzero(component)) / skeleton_pixels)


def _looks_like_skeleton(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return True
    return np.array_equal(mask, morphology.skeletonize(mask))


def _endpoint_branch_length(skeleton: np.ndarray, start_y: int, start_x: int, limit: int) -> int:
    previous: tuple[int, int] | None = None
    current = (start_y, start_x)
    length = 0

    while length < limit:
        neighbors = _skeleton_neighbors(skeleton, *current)
        if previous is not None and len(neighbors) != 2:
            return length

        candidates = [neighbor for neighbor in neighbors if neighbor != previous]
        if len(candidates) != 1:
            return length

        previous = current
        current = candidates[0]
        length += 1

    return length


def _skeleton_neighbors(skeleton: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    height, width = skeleton.shape
    for yy in range(max(0, y - 1), min(height, y + 2)):
        for xx in range(max(0, x - 1), min(width, x + 2)):
            if (yy, xx) != (y, x) and skeleton[yy, xx]:
                neighbors.append((int(yy), int(xx)))
    return neighbors


def _clamp_int(value: str | None, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(float(value))
    except ValueError:
        return default
    return max(low, min(high, parsed))


def _clamp_float(value: str | None, default: float, low: float, high: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(low, min(high, parsed))
