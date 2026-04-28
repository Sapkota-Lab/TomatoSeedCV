import argparse
from pathlib import Path

import cv2

from train_model import segment_seeds, summarize_seeds, annotate, describe


def main():
    """
    Main pipeline to segment seeds, measure their sizes, and generate outputs.

    TODO:
    - Run with test image to verify brown seed segmentation works
    - Calibrate mm_per_pixel using a known reference object or ruler
    - Adjust --mm-per-pixel argument once calibration is complete
    """
    parser = argparse.ArgumentParser(description="Detect tomato seeds and estimate sizes.")
    parser.add_argument("--image", required=True, help="Path to an input image (e.g., Images/img006.jpg)")
    parser.add_argument(
        "--has-ruler",
        action="store_true",
        default=False,
        help="Set this flag if the image contains a ruler for automatic calibration.",
    )
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=0.005341,
        help="Manual calibration value. If --has-ruler is set, this will be auto-calibrated if possible.",
    )
    parser.add_argument("--min-area-px", type=float, default=20.0, help="Pixel area threshold to discard tiny blobs.")
    parser.add_argument("--min-area-mm2", type=float, default=2.0, help="Minimum seed area in mm² (only applied if calibrated).")
    parser.add_argument("--max-area-mm2", type=float, default=10.0, help="Maximum seed area in mm² to filter merged clusters.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store mask and overlay images.")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Use manual calibration only; ignore the ruler entirely.
    mm_per_pixel = args.mm_per_pixel

    mask, seeds = segment_seeds(image, min_area_px=args.min_area_px)
    summary = summarize_seeds(seeds, mm_per_pixel=mm_per_pixel)

    # Apply optional area filters (mm²) if calibration is available
    if mm_per_pixel and mm_per_pixel > 0:
        filtered_summary = []
        for seed in summary:
            if seed['area_mm2'] is not None:
                if args.min_area_mm2 <= seed['area_mm2'] <= args.max_area_mm2:
                    filtered_summary.append(seed)
            else:
                # If no calibration, keep seed
                filtered_summary.append(seed)
        summary = filtered_summary

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.jpg"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), annotate(image, summary))

    describe(summary)
    print(f"Saved mask to {mask_path}")
    print(f"Saved overlay to {overlay_path}")


if __name__ == "__main__":
    main()
