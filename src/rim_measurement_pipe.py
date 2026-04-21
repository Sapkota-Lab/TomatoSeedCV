import os
import csv
import time

from roboflow_rimdetect import run_rim_detection, summarize_rim, mm_per_pixel

INPUT_FOLDER = "../Images/bisected_IN"
OUTPUT_FILE = "../outputs/bisected_rim_measurements.csv"


def process_images():
    results = []
    failed = []
    request_count = 0

    print("Entered process_images()")
    print(f"Looking in: {os.path.abspath(INPUT_FOLDER)}")

    # Count total images first
    total_images = 0
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                total_images += 1

    print(f"Total images found: {total_images}")

    # Process images
    for root, dirs, files in os.walk(INPUT_FOLDER):
        print(f"Checking folder: {root} ({len(files)} files)")
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, filename)

                try:
                    request_count += 1
                    print(f"[{request_count}/{total_images}] Processing {image_path}")

                    detection = run_rim_detection(image_path)
                    summary = summarize_rim(detection["mask"], mm_per_pixel)

                    # Get folder and image name
                    folder_name = os.path.basename(root)

                    # Build label
                    label = f"{folder_name} - {filename}"

                    results.append({
                        "image_name": label,
                        "avg_thickness_mm": summary["avg_thickness_mm"],
                        "max_thickness_mm": summary["max_thickness_mm"],
                        "min_thickness_mm": summary["min_thickness_mm"],
                    })

                    time.sleep(0.2)  # avoid API rate limiting

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    failed.append(image_path)

    print(f"\nFinished loop.")
    print(f"Processed successfully: {len(results)} / {total_images}")
    print(f"Failed images: {len(failed)}")

    return results


def save_to_csv(results):
    print("Entered save_to_csv()")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "avg_thickness_mm",
                "max_thickness_mm",
                "min_thickness_mm"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    print("Starting batch rim measurement script...")
    results = process_images()
    save_to_csv(results)