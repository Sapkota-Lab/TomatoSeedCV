import pathlib as path
import sys
import csv

import cv2
from train_model import annotate, describe, segment_seeds, summarize_seeds

def create_results_dir():
    results_folder_path = path.Path("results")
    results_folder_path.mkdir(exist_ok=True)
    masks_folder_path = results_folder_path / "masks"
    overlays_folder_path = results_folder_path / "overlays"
    masks_folder_path.mkdir(exist_ok=True)
    overlays_folder_path.mkdir(exist_ok=True)
    return masks_folder_path, overlays_folder_path


def save_results(image_path, mask, overlay):
    masks_folder_path, overlays_folder_path = create_results_dir()
    mask_path = masks_folder_path / f"{image_path.stem}_mask.png"
    overlay_path = overlays_folder_path / f"{image_path.stem}_overlay.jpg"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Saved mask to {mask_path}")
    print(f"Saved overlay to {overlay_path}")


def build_summary_rows(image_path, summary):
    rows = []
    for idx, seed in enumerate(summary, start=1):
        rows.append(
            {
                "image": image_path.name,
                "seed_id": idx,
                "area_px": seed["area_px"],
                "perimeter_px": seed["perimeter_px"],
                "major_px": seed["major_px"],
                "minor_px": seed["minor_px"],
                "eq_diam_px": seed["eq_diam_px"],
                "aspect_ratio": seed["aspect_ratio"],
                "circularity": seed["circularity"],
                "elongation": seed["elongation"],
                "compactness": seed["compactness"],
                "roundness": seed["roundness"],
                "area_mm2": seed["area_mm2"],
                "perimeter_mm": seed["perimeter_mm"],
                "major_mm": seed["major_mm"],
                "minor_mm": seed["minor_mm"],
                "eq_diam_mm": seed["eq_diam_mm"],
            }
        )
    return rows


def save_summary_csv(rows):
    results_folder_path = path.Path("results")
    results_folder_path.mkdir(exist_ok=True)
    summary_path = results_folder_path / "summary.csv"

    fieldnames = [
        "image",
        "seed_id",
        "area_px",
        "perimeter_px",
        "major_px",
        "minor_px",
        "eq_diam_px",
        "aspect_ratio",
        "circularity",
        "elongation",
        "compactness",
        "roundness",
        "area_mm2",
        "perimeter_mm",
        "major_mm",
        "minor_mm",
        "eq_diam_mm",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-seed summary to {summary_path}")

def main():
    if len(sys.argv) != 2:
        print("Please provide the path to the folder containing the cropped images as a command-line argument")
        sys.exit(1)

    cropped_images_folder_path = path.Path(sys.argv[1])
    print(f"Cropped images folder path: {cropped_images_folder_path}")

    all_rows = []

    for image_path in cropped_images_folder_path.glob("*.jpg"):
        print(f"Processing {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image at {image_path}, skipping.")
            continue
        mask, seeds = segment_seeds(image, min_area_px=20.0)
        summary = summarize_seeds(seeds, mm_per_pixel=0.005341)
        overlay = annotate(image, summary, mm_per_pixel=0.005341)
        describe(summary)
        all_rows.extend(build_summary_rows(image_path, summary))
        save_results(image_path, mask, overlay)

    save_summary_csv(all_rows)


if __name__ == "__main__":
    main()
    