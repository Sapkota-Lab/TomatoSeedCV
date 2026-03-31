import math
import cv2
import numpy as np
    
def segment_seeds(image_bgr: np.ndarray, min_area_px: float = 20.0):

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian blur to smooth texture and blend dark spots inside seeds
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Apply binary thresholding (Otsu's method for automatic threshold selection)
    # Regular BINARY threshold: foreground (seeds) = white (255), background = black (0)
    _, threshold_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3.5: Apply morphological operations to remove noise
    # Opening removes small white noise in the background
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Closing fills small holes inside seeds
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Step 4: Find external contours on the white seeds
    # Use CHAIN_APPROX_NONE for accurate perimeter measurement
    contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    seeds = []
    for contour in contours:
        # Step 5: Filter by area - only keep contours larger than min_area_px
        area_px = cv2.contourArea(contour)
        
        # Only keep seeds larger than the minimum area threshold
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
    
    # Return inverted mask (black seeds on white background) for display
    display_mask = cv2.bitwise_not(threshold_mask)
    return display_mask, seeds

def summarize_seeds(seeds, mm_per_pixel: float | None):
    summary = []
    for seed in seeds:
        major_px = max(seed["width_px"], seed["height_px"])
        minor_px = min(seed["width_px"], seed["height_px"])
        eq_diam_px = math.sqrt(4.0 * seed["area_px"] / math.pi)
        perimeter_px = cv2.arcLength(seed["contour"], True)  # Full perimeter, no artificial scaling

        def to_mm(value_px: float) -> float | None:
            # TODO: Replace with actual calibration value once mm_per_pixel is determined
            return value_px * mm_per_pixel if mm_per_pixel else None

        # Shape indices (dimensionless)
        aspect_ratio = major_px / minor_px if minor_px > 0 else 0
        circularity = (4 * math.pi * seed["area_px"]) / (perimeter_px ** 2) if perimeter_px > 0 else 0
        elongation = 1 - (minor_px / major_px) if major_px > 0 else 0
        compactness = (perimeter_px ** 2) / seed["area_px"] if seed["area_px"] > 0 else 0
        roundness = (4 * seed["area_px"]) / (math.pi * major_px ** 2) if major_px > 0 else 0

        summary.append(
            {
                "contour": seed["contour"],
                "area_px": seed["area_px"],
                "perimeter_px": perimeter_px,
                "major_px": major_px,
                "minor_px": minor_px,
                "eq_diam_px": eq_diam_px,
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "elongation": elongation,
                "compactness": compactness,
                "roundness": roundness,
                # TODO: These will be None until mm_per_pixel is calibrated
                "area_mm2": seed["area_px"] * (mm_per_pixel**2) if mm_per_pixel else None,
                "perimeter_mm": to_mm(perimeter_px),
                "major_mm": to_mm(major_px),
                "minor_mm": to_mm(minor_px),
                "eq_diam_mm": to_mm(eq_diam_px),
            }
        )

    return summary

def annotate(image_bgr: np.ndarray, seed_metrics, mm_per_pixel: float | None):
    annotated = image_bgr.copy()
    color = (0, 180, 255)

    for idx, seed in enumerate(seed_metrics, start=1):
        contour = seed["contour"]
        cv2.drawContours(annotated, [contour], -1, color, 1)

        x, y, w, h = cv2.boundingRect(contour)

        # Show ID and area (prefer calibrated mm^2 when available)
        if seed.get("area_mm2") is not None:
            area_text = f"{seed['area_mm2']:.2f} mm^2"
        else:
            area_text = f"{seed['area_px']:.1f} px^2"
        label = f"ID {idx} | A {area_text}"

        cv2.putText(
            annotated,
            label,
            (x, max(0, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated

def describe(summary):
    if not summary:
        print("No seeds detected after filtering.")
        return

    eq_diams_px = [s["eq_diam_px"] for s in summary]
    eq_diams_mm = [s["eq_diam_mm"] for s in summary if s["eq_diam_mm"] is not None]
    perimeters_px = [s["perimeter_px"] for s in summary]
    perimeters_mm = [s["perimeter_mm"] for s in summary if s["perimeter_mm"] is not None]
    areas_px = [s["area_px"] for s in summary]
    areas_mm = [s["area_mm2"] for s in summary if s["area_mm2"] is not None]
    aspect_ratios = [s["aspect_ratio"] for s in summary]
    circularities = [s["circularity"] for s in summary]
    elongations = [s["elongation"] for s in summary]
    compactnesses = [s["compactness"] for s in summary]
    roundnesses = [s["roundness"] for s in summary]

    def pct(values, p):
        if not values:
            return None
        idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
        return sorted(values)[idx]

    print(f"Seeds kept: {len(summary)}")
    print(f"Equivalent diameter px | median={pct(eq_diams_px, 50):.2f}  p10={pct(eq_diams_px, 10):.2f}  p90={pct(eq_diams_px, 90):.2f}")
    print(f"Perimeter px | median={pct(perimeters_px, 50):.2f}  p10={pct(perimeters_px, 10):.2f}  p90={pct(perimeters_px, 90):.2f}")
    print(f"Area px^2 | median={pct(areas_px, 50):.2f}  p10={pct(areas_px, 10):.2f}  p90={pct(areas_px, 90):.2f}")
    print(f"Aspect Ratio | median={pct(aspect_ratios, 50):.3f}  p10={pct(aspect_ratios, 10):.3f}  p90={pct(aspect_ratios, 90):.3f}")
    print(f"Circularity | median={pct(circularities, 50):.3f}  p10={pct(circularities, 10):.3f}  p90={pct(circularities, 90):.3f}")
    print(f"Elongation | median={pct(elongations, 50):.3f}  p10={pct(elongations, 10):.3f}  p90={pct(elongations, 90):.3f}")
    print(f"Compactness | median={pct(compactnesses, 50):.3f}  p10={pct(compactnesses, 10):.3f}  p90={pct(compactnesses, 90):.3f}")
    print(f"Roundness | median={pct(roundnesses, 50):.3f}  p10={pct(roundnesses, 10):.3f}  p90={pct(roundnesses, 90):.3f}")

    if eq_diams_mm:
        print(f"Equivalent diameter mm | median={pct(eq_diams_mm, 50):.3f}  p10={pct(eq_diams_mm, 10):.3f}  p90={pct(eq_diams_mm, 90):.3f}")
    if perimeters_mm:
        print(f"Perimeter mm | median={pct(perimeters_mm, 50):.3f}  p10={pct(perimeters_mm, 10):.3f}  p90={pct(perimeters_mm, 90):.3f}")
    if areas_mm:
        print(f"Area mm^2 | median={pct(areas_mm, 50):.3f}  p10={pct(areas_mm, 10):.3f}  p90={pct(areas_mm, 90):.3f}")
