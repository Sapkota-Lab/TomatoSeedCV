"""
Roboflow-backed whole-seed segmentation helpers for the Shiny app.

The app expects each seed pipeline to return an original image, binary mask,
annotated overlay, and per-seed measurements. This module adapts the Roboflow
instance-segmentation response into the same contour-based shape used by the
local OpenCV pipeline in train_model.py.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

try:
    from src.train_model import summarize_seeds
except ImportError:  # Allows direct execution from inside src/
    from train_model import summarize_seeds


env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(env_path, override=True)

DEFAULT_MODEL_ID = "lab-seed-detection/13"
DEFAULT_API_URL = "https://outline.roboflow.com"
DEFAULT_CONFIDENCE = 0.20
DEFAULT_API_CONFIDENCE = 40
DEFAULT_API_OVERLAP = 30


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def call_roboflow(image_path: str, model_id: str) -> dict[str, Any]:
    """Call the Roboflow hosted image endpoint used by the original script."""
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Add it to TomatoSeedCV/.env before running "
            "whole-seed Roboflow detection."
        )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    if not ok:
        raise ValueError("Failed to JPEG-encode image for Roboflow.")

    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    api_url = os.getenv("WHOLE_SEED_API_URL", DEFAULT_API_URL).rstrip("/")
    confidence = int(_get_float_env("WHOLE_SEED_API_CONFIDENCE", DEFAULT_API_CONFIDENCE))
    overlap = int(_get_float_env("WHOLE_SEED_API_OVERLAP", DEFAULT_API_OVERLAP))
    url = (
        f"{api_url}/{model_id}"
        f"?api_key={api_key}&confidence={confidence}&overlap={overlap}"
    )

    response = requests.post(
        url,
        data=image_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def extract_mask_from_predictions(
    result: dict[str, Any],
    image_shape: tuple[int, ...],
    confidence_threshold: float | None = None,
) -> np.ndarray:
    """Build a binary foreground mask from Roboflow polygon or box predictions."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    predictions = result.get("predictions", [])
    min_confidence = (
        DEFAULT_CONFIDENCE if confidence_threshold is None else confidence_threshold
    )

    for pred in predictions:
        confidence = float(pred.get("confidence", 1.0))
        if confidence < min_confidence:
            continue

        points = pred.get("points") or []
        if len(points) >= 3:
            polygon = np.array(
                [
                    [
                        int(np.clip(round(point["x"]), 0, w - 1)),
                        int(np.clip(round(point["y"]), 0, h - 1)),
                    ]
                    for point in points
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [polygon], 255)
            continue

        x, y, box_w, box_h = (pred.get(key) for key in ("x", "y", "width", "height"))
        if None in (x, y, box_w, box_h):
            continue

        x1 = max(0, int(round(x - box_w / 2)))
        y1 = max(0, int(round(y - box_h / 2)))
        x2 = min(w - 1, int(round(x + box_w / 2)))
        y2 = min(h - 1, int(round(y + box_h / 2)))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    if np.count_nonzero(mask) == 0:
        return mask

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    return mask


def prediction_image_shape(
    result: dict[str, Any],
    fallback_shape: tuple[int, ...],
) -> tuple[int, int]:
    """Return the height/width Roboflow used for prediction coordinates."""
    image_info = result.get("image") or {}
    width = image_info.get("width")
    height = image_info.get("height")

    if width and height:
        return int(height), int(width)

    return fallback_shape[:2]


def scale_mask_to_image(mask: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Resize a prediction-coordinate mask to the uploaded image dimensions."""
    target_h, target_w = image_shape[:2]
    if mask.shape[:2] == (target_h, target_w):
        return mask

    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def seeds_from_mask(mask: np.ndarray, min_area_px: float = 20.0) -> list[dict[str, Any]]:
    """Convert a binary whole-seed mask into contour records for measurement."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    seeds = []
    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < min_area_px:
            continue

        rect = cv2.minAreaRect(contour)
        width_px, height_px = rect[1]
        if width_px == 0 or height_px == 0:
            continue

        seeds.append(
            {
                "contour": contour,
                "area_px": area_px,
                "width_px": float(width_px),
                "height_px": float(height_px),
                "center": tuple(rect[0]),
                "angle": float(rect[2]),
            }
        )

    seeds.sort(key=lambda seed: seed["area_px"], reverse=True)
    return seeds


def filter_summary_by_area(
    summary: list[dict[str, Any]],
    min_area_mm2: float | None,
    max_area_mm2: float | None,
) -> list[dict[str, Any]]:
    """Apply the Shiny app's optional calibrated area filter."""
    if min_area_mm2 is None and max_area_mm2 is None:
        return summary

    filtered = []
    for seed in summary:
        area_mm2 = seed.get("area_mm2")
        if area_mm2 is None:
            filtered.append(seed)
            continue

        above_min = min_area_mm2 is None or area_mm2 >= min_area_mm2
        below_max = max_area_mm2 is None or area_mm2 <= max_area_mm2
        if above_min and below_max:
            filtered.append(seed)

    return filtered


def mask_from_summary(image_shape: tuple[int, ...], summary: list[dict[str, Any]]) -> np.ndarray:
    """Create the displayed mask from the filtered seed contours."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    contours = [seed["contour"] for seed in summary if seed.get("contour") is not None]
    if contours:
        cv2.fillPoly(mask, contours, 255)
    return mask


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Show original image pixels inside the seed mask on a white background."""
    masked_image = np.full_like(image, 255)
    masked_image[mask > 0] = image[mask > 0]
    return masked_image


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    summary: list[dict[str, Any]],
    mm_per_pixel: float | None,
) -> np.ndarray:
    """Create a visible overlay for high-resolution whole-seed images."""
    overlay_layer = image.copy()
    overlay_layer[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay_layer, 0.35, image, 0.65, 0)

    h, w = image.shape[:2]
    max_dim = max(h, w)
    contour_thickness = max(3, int(round(max_dim / 350)))
    text_scale = max(0.8, max_dim / 2500)
    text_thickness = max(2, int(round(max_dim / 1200)))
    label_padding = max(6, int(round(max_dim / 700)))

    for idx, seed in enumerate(summary, start=1):
        contour = seed["contour"]
        cv2.drawContours(
            overlay,
            [contour],
            -1,
            (0, 0, 255),
            contour_thickness,
            cv2.LINE_AA,
        )

        x, y, _, _ = cv2.boundingRect(contour)
        if seed.get("area_mm2") is not None:
            area_text = f"{seed['area_mm2']:.2f} mm^2"
        else:
            area_text = f"{seed['area_px']:.1f} px^2"
        label = f"{idx}: {area_text}"

        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )
        label_x = max(0, min(x, w - label_w - label_padding * 2))
        label_y = max(label_h + label_padding * 2, y - label_padding)

        cv2.rectangle(
            overlay,
            (label_x, label_y - label_h - label_padding),
            (label_x + label_w + label_padding * 2, label_y + baseline + label_padding),
            (255, 255, 255),
            thickness=-1,
        )
        cv2.putText(
            overlay,
            label,
            (label_x + label_padding, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return overlay


def run_whole_seed_detection(
    image_path: str,
    min_area_px: float = 20.0,
    mm_per_pixel: float | None = None,
    min_area_mm2: float | None = None,
    max_area_mm2: float | None = None,
    model_id: str | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run Roboflow whole-seed segmentation and return Shiny-ready outputs.

    Returns:
        A dict with original image, filtered binary mask, annotated overlay,
        per-seed summary, and raw Roboflow result.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    model = model_id or os.getenv("WHOLE_SEED_MODEL_ID", DEFAULT_MODEL_ID)
    confidence = (
        _get_float_env("WHOLE_SEED_CONFIDENCE", DEFAULT_CONFIDENCE)
        if confidence_threshold is None
        else confidence_threshold
    )

    result = call_roboflow(image_path, model)
    prediction_shape = prediction_image_shape(result, image.shape)
    prediction_mask = extract_mask_from_predictions(result, prediction_shape, confidence)
    raw_mask = scale_mask_to_image(prediction_mask, image.shape)
    seeds = seeds_from_mask(raw_mask, min_area_px=min_area_px)
    unfiltered_summary = summarize_seeds(seeds, mm_per_pixel=mm_per_pixel)
    summary = unfiltered_summary

    if mm_per_pixel and mm_per_pixel > 0:
        summary = filter_summary_by_area(summary, min_area_mm2, max_area_mm2)

    area_filter_removed_all = bool(unfiltered_summary and not summary)
    if area_filter_removed_all:
        summary = unfiltered_summary

    mask = mask_from_summary(image.shape, summary)
    masked_image = apply_mask_to_image(image, mask)
    overlay = create_overlay(image, mask, summary, mm_per_pixel)

    return {
        "original": image,
        "mask": masked_image,
        "binary_mask": mask,
        "overlay": overlay,
        "summary": summary,
        "raw_mask": raw_mask,
        "raw_result": result,
        "diagnostics": {
            "prediction_count": len(result.get("predictions", [])),
            "unfiltered_seed_count": len(unfiltered_summary),
            "displayed_seed_count": len(summary),
            "area_filter_removed_all": area_filter_removed_all,
            "prediction_shape": prediction_shape,
            "image_shape": image.shape[:2],
        },
    }
