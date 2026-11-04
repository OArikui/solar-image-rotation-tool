"""Tests for image preview conversion and image loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from solar_rotation_tool.image_display import to_rgb
from solar_rotation_tool.image_io import load_image


class ImageUtilityTests(unittest.TestCase):
    """Test science-image loading and display-only normalization."""

    def test_display_conversion_normalizes_to_rgb(self) -> None:
        """A finite grayscale image becomes a normalized three-channel preview."""
        preview = to_rgb(np.asarray([[0.0, 5.0], [10.0, 15.0]]))
        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertEqual(int(preview[0, 0, 0]), 0)
        self.assertEqual(int(preview[1, 1, 0]), 255)

    def test_display_conversion_handles_non_finite_and_constant_values(self) -> None:
        """Non-finite values are safe and constant images remain visible."""
        preview = to_rgb(np.asarray([[np.nan, 5.0], [5.0, np.inf]]))
        self.assertTrue(np.isfinite(preview).all())
        self.assertEqual(int(preview[0, 1, 0]), 127)

    def test_load_image_reads_tiff_without_modifying_pixel_values(self) -> None:
        """TIFF input is returned as a two-dimensional array with source values."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "solar.tif"
            source = np.asarray([[0, 100], [200, 65535]], dtype=np.uint16)
            tifffile.imwrite(path, source)
            loaded = load_image(path)
            np.testing.assert_array_equal(loaded, source)

    def test_load_image_rejects_unsupported_formats(self) -> None:
        """Unsupported filename extensions are rejected before file decoding."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "solar.bmp"
            path.write_bytes(b"placeholder")
            with self.assertRaises(ValueError):
                load_image(path)


if __name__ == "__main__":
    unittest.main()
