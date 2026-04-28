"""Manual helper for previewing the whole-seed mask overlay."""

from pathlib import Path

import cv2

from src.train_model import annotate, segment_seeds


IMAGE_PATH = Path("Images/whole_imgs/img003.jpg")


def main() -> None:
    """Open a local image and show its annotated segmentation overlay."""
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    _, seeds = segment_seeds(image)
    overlay = annotate(image, seeds)

    cv2.imshow("Seed Mask Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
