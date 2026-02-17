import cv2
import numpy as np


def detect_ruler_presence(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(60, int(min(gray.shape[:2]) * 0.2)),
        maxLineGap=8,
    )
    if lines is None:
        return False

    long_lines = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= max(80, int(min(gray.shape[:2]) * 0.25)):
            long_lines += 1
    return long_lines >= 2


def extract_ruler_calibration(image_bgr: np.ndarray, has_ruler: bool = True) -> float | None:
    if not has_ruler:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Emphasize ruler tick marks (dark on light background typical)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    # Extract thin vertical tick-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    ticks = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(ticks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 6 or w > 6:
            continue
        centers.append((x + w / 2.0, y + h / 2.0))

    if len(centers) < 5:
        return None

    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])

    # Determine ruler orientation based on spread
    if np.std(xs) > np.std(ys):
        axis_vals = xs
    else:
        axis_vals = ys

    axis_vals = np.sort(axis_vals)
    diffs = np.diff(axis_vals)
    diffs = diffs[(diffs > 2) & (diffs < np.percentile(diffs, 90))]
    if diffs.size < 3:
        return None

    px_per_cm = float(np.median(diffs))
    if px_per_cm <= 0:
        return None
    return 10.0 / px_per_cm


def calibrate_from_reference_object(
    image_bgr: np.ndarray,
    reference_length_mm: float,
    reference_bounding_box: tuple | None = None
) -> float | None:
    if reference_length_mm <= 0:
        return None

    if reference_bounding_box is not None:
        _, _, w, h = reference_bounding_box
        length_px = max(w, h)
        if length_px <= 0:
            return None
        return reference_length_mm / float(length_px)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    length_px = max(w, h)
    if length_px <= 0:
        return None
    return reference_length_mm / float(length_px)
