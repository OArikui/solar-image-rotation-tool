"""Read grayscale TIFF and FITS solar images."""

from pathlib import Path
from typing import Any

from astropy.io import fits
import numpy as np
import tifffile

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".fits", ".fit", ".fts"}


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a 2D grayscale TIFF/FITS image as a NumPy array."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {suffix}")
    if suffix in {".tif", ".tiff"}:
        image: Any = tifffile.imread(path)
    else:
        image = fits.getdata(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"Expected a non-empty 2D grayscale image, got shape {array.shape}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
