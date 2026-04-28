import os

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(env_path, override=True)  # Ensure we load the .env file

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

MODEL_ID = "seed-rim-detection/7"

MM_PER_PIXEL = 0.00811


def run_rim_detection(image_path):
    if CLIENT is None:
        raise RuntimeError(
            "inference-sdk is not installed for this Python version. "
            "Use Python 3.12 for bisected-seed rim detection, or install a "
            "compatible inference-sdk build."
        )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")

    result = CLIENT.infer(image_path, model_id=MODEL_ID)

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    predictions = result.get("predictions", [])  # Roboflow returns JSON points

    # For all points the API returns, we fill in to make the mask
    for pred in predictions:
        points = pred.get("points", [])
        if len(points) >= 3:
            polygon = np.array(
                [[int(p["x"]), int(p["y"])] for p in points],
                dtype=np.int32
            )
            cv2.fillPoly(mask, [polygon], 255)

    # Distance transform calculates based on distance from each foreground pixel
    # to the closest background pixel.
    # Max will be the middle pixel distance * 2 to account for each side, giving us total max width.

    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

    return {
        "mask": mask,
        "overlay": overlay,
        "raw_result": result,
    }

def summarize_rim(mask, calibration_mm_per_pixel):


    rim_area_px = int(np.sum(mask > 0))

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    rim_pixels = dist_transform[mask > 0]

    if rim_pixels.size > 0:
        avg_thickness_px = float(np.mean(rim_pixels) * 2)
        max_thickness_px = float(np.max(rim_pixels) * 2)
        min_thickness_px = float(np.min(rim_pixels) * 2)
        std_thickness_px = float(np.std(rim_pixels) * 2)

    else:
        avg_thickness_px = 0.0
        max_thickness_px = 0.0
        min_thickness_px = 0.0
        std_thickness_px = 0.0

    summary = {
        "rim_area_px": rim_area_px,
        "avg_thickness_px": avg_thickness_px,
        "max_thickness_px": max_thickness_px,
        "min_thickness_px": min_thickness_px,
        "std_thickness_px": std_thickness_px,
    }

    if calibration_mm_per_pixel is not None and calibration_mm_per_pixel > 0:
        summary.update({
            "rim_area_mm2": rim_area_px * (calibration_mm_per_pixel ** 2),
            "avg_thickness_mm": avg_thickness_px * calibration_mm_per_pixel,
            "max_thickness_mm": max_thickness_px * calibration_mm_per_pixel,
            "min_thickness_mm": min_thickness_px * calibration_mm_per_pixel,
            "std_thickness_mm": std_thickness_px * calibration_mm_per_pixel,
        })
    else:
        summary.update({
            "rim_area_mm2": None,
            "avg_thickness_mm": None,
            "max_thickness_mm": None,
            "min_thickness_mm": None,
            "std_thickness_mm": None,
        })

    return summary
