"""Convert scientific grayscale arrays to display-ready RGB images."""

import cv2
import numpy as np


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D image to 8-bit RGB for display.

    Each source image is normalized independently using its finite minimum and
    maximum. This is intended for visual comparison, not quantitative analysis.
    """
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Display image must be a non-empty 2D array")
    values = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        gray = np.zeros(image.shape, dtype=np.uint8)
    else:
        minimum = float(values[finite].min())
        maximum = float(values[finite].max())
        gray = np.zeros(image.shape, dtype=np.uint8)
        if maximum > minimum:
            scaled = (values[finite] - minimum) / (maximum - minimum)
            gray[finite] = np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            gray[finite] = 127
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
