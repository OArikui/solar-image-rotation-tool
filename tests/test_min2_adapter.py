"""Integration tests for MIN2 solar-disk detection."""

import unittest

import numpy as np

from solar_rotation_tool.min2_adapter import detect_solar_disk


class Min2AdapterTests(unittest.TestCase):
    """Verify that the bundled MIN2 detector can find a synthetic solar disk."""

    def _synthetic_disk(self) -> np.ndarray:
        height, width = 420, 440
        expected_center = (218.0, 204.0)
        expected_radius = 145.0
        y, x = np.ogrid[:height, :width]
        distance_squared = (x - expected_center[0]) ** 2 + (y - expected_center[1]) ** 2
        image = np.zeros((height, width), dtype=np.uint8)
        image[distance_squared <= expected_radius**2] = 230
        return image

    def test_detects_center_and_radius_using_bundled_config(self) -> None:
        """The bundled config should recover a clean bright disk."""
        disk = detect_solar_disk(self._synthetic_disk())

        self.assertAlmostEqual(disk.center[0], 218.0, delta=3.0)
        self.assertAlmostEqual(disk.center[1], 204.0, delta=3.0)
        self.assertAlmostEqual(disk.radius, 145.0, delta=3.0)
        self.assertEqual(disk.method, "MIN2 v2.4.3")

    def test_explicit_parameters_override_config(self) -> None:
        """Explicit detector arguments remain supported after adding YAML config."""
        disk = detect_solar_disk(
            self._synthetic_disk(),
            n=12,
            light_threshold=50,
            limb_wigth=24,
            iter_cycles=1,
        )

        self.assertEqual(disk.method, "MIN2 v2.4.3")


if __name__ == "__main__":
    unittest.main()
