"""
Train a model for seed detection and measure seed properties.

This module contains functions for training machine learning models
to detect and segment strawberry seeds, as well as functions for 
measuring seed sizes and properties.
"""

import math

import cv2
import numpy as np


def train_seed_detection_model(training_data, training_labels):
    """
    Train a seed detection model using provided training data.
    
    Args:
        training_data: Array of training images
        training_labels: Array of corresponding seed masks or labels
        
    Returns:
        Trained model object
        
    TODO: Implement ML model training (e.g., using TensorFlow, PyTorch, or sklearn)
    """
    pass


def load_seed_detection_model(model_path: str):
    """
    Load a pre-trained seed detection model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
        
    TODO: Implement model loading logic
    """
    pass


def predict_seed_mask(model, image: np.ndarray) -> np.ndarray:
    """
    Use a trained model to predict seed mask from an image.
    
    Args:
        model: Trained seed detection model
        image: Input image (BGR)
        
    Returns:
        Binary mask of detected seeds
        
    TODO: Implement prediction using the trained model
    """
    pass


def segment_seeds(image_bgr: np.ndarray, min_area_px: float = 20.0):
    """
    Segment brown seeds from white background using color thresholding and morphology.
    
    TODO: Calibrate HSV thresholds for brown seeds on white background.
    Current values are placeholders and need to be tuned based on your image.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # TODO: Adjust these thresholds for brown-ish seeds
    # Brown hues are typically in range 10-20 (HSV H channel)
    lower = np.array([10, 50, 50], dtype=np.uint8)
    upper = np.array([20, 255, 200], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    seeds.sort(key=lambda s: s["area_px"], reverse=True)
    return mask, seeds


def summarize_seeds(seeds, mm_per_pixel: float | None):
    """
    Compute per-seed metrics and unit conversions.
    
    TODO: Implement pixel-to-mm calibration.
    Measure a reference object (e.g., a ruler) in the image and calculate mm_per_pixel.
    """
    summary = []
    for seed in seeds:
        major_px = max(seed["width_px"], seed["height_px"])
        minor_px = min(seed["width_px"], seed["height_px"])
        eq_diam_px = math.sqrt(4.0 * seed["area_px"] / math.pi)

        def to_mm(value_px: float) -> float | None:
            # TODO: Replace with actual calibration value once mm_per_pixel is determined
            return value_px * mm_per_pixel if mm_per_pixel else None

        summary.append(
            {
                "contour": seed["contour"],
                "area_px": seed["area_px"],
                "major_px": major_px,
                "minor_px": minor_px,
                "eq_diam_px": eq_diam_px,
                # TODO: These will be None until mm_per_pixel is calibrated
                "area_mm2": seed["area_px"] * (mm_per_pixel**2) if mm_per_pixel else None,
                "major_mm": to_mm(major_px),
                "minor_mm": to_mm(minor_px),
                "eq_diam_mm": to_mm(eq_diam_px),
            }
        )

    return summary


def annotate(image_bgr: np.ndarray, seed_metrics, mm_per_pixel: float | None):
    """Draw contours and a short size label on top of each seed."""
    annotated = image_bgr.copy()
    color = (0, 180, 255)

    for idx, seed in enumerate(seed_metrics, start=1):
        contour = seed["contour"]
        cv2.drawContours(annotated, [contour], -1, color, 1)

        x, y, w, h = cv2.boundingRect(contour)
        label = f"{seed['eq_diam_px']:.1f}px"
        if seed["eq_diam_mm"] is not None:
            label = f"{seed['eq_diam_mm']:.2f}mm"

        cv2.putText(
            annotated,
            label,
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def describe(summary):
    """Print a compact textual report."""
    if not summary:
        print("No seeds detected after filtering.")
        return

    eq_diams_px = [s["eq_diam_px"] for s in summary]
    eq_diams_mm = [s["eq_diam_mm"] for s in summary if s["eq_diam_mm"] is not None]

    def pct(values, p):
        if not values:
            return None
        idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
        return sorted(values)[idx]

    print(f"Seeds kept: {len(summary)}")
    print(f"Equivalent diameter px | median={pct(eq_diams_px, 50):.2f}  p10={pct(eq_diams_px, 10):.2f}  p90={pct(eq_diams_px, 90):.2f}")

    if eq_diams_mm:
        print(f"Equivalent diameter mm | median={pct(eq_diams_mm, 50):.3f}  p10={pct(eq_diams_mm, 10):.3f}  p90={pct(eq_diams_mm, 90):.3f}")
