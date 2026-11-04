"""Tests for two-image rotation and similarity measurements."""

import unittest

import numpy as np

from solar_rotation_tool.rotation_measurement import (
    add_rotation_about_center,
    calculate_similarity_alignment,
    center_radius_alignment_matrix,
    centered_fit_matrix,
)


class RotationMeasurementTests(unittest.TestCase):
    """Verify corresponding-point transforms and image-coordinate angle signs."""

    def test_identity_alignment(self) -> None:
        """An image aligned with itself has zero rotation and unit scale."""
        points = [(10.0, 20.0), (40.0, 20.0)]
        alignment = calculate_similarity_alignment(points, points)
        np.testing.assert_allclose(
            alignment.matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-12
        )
        self.assertAlmostEqual(alignment.rotation_deg, 0.0)
        self.assertAlmostEqual(alignment.scale, 1.0)

    def test_rotation_scale_and_translation(self) -> None:
        """The transform recovers rotation, scale and translation."""
        comparison = np.asarray([[0.0, 0.0], [10.0, 0.0]])
        reference = np.asarray([[5.0, 7.0], [5.0, 27.0]])
        alignment = calculate_similarity_alignment(reference, comparison)
        self.assertAlmostEqual(alignment.rotation_deg, 90.0)
        self.assertAlmostEqual(alignment.scale, 2.0)
        np.testing.assert_allclose(
            alignment.transform_points(comparison), reference, atol=1e-12
        )

    def test_clockwise_screen_rotation_is_positive(self) -> None:
        """Image-coordinate angle signs follow x-right/y-down coordinates."""
        comparison = [(0.0, 0.0), (0.0, 10.0)]
        reference = [(2.0, 3.0), (12.0, 3.0)]
        alignment = calculate_similarity_alignment(reference, comparison)
        self.assertAlmostEqual(alignment.rotation_deg, -90.0)
        np.testing.assert_allclose(
            alignment.transform_points(comparison), reference, atol=1e-12
        )

    def test_centered_fit_preserves_aspect_ratio(self) -> None:
        """Initial placement centers the image without distortion."""
        matrix = centered_fit_matrix((100, 200), (50, 50))
        np.testing.assert_allclose(matrix, [[2.0, 0, 50.0], [0, 2.0, 0]])

    def test_rotation_is_composed_around_reference_center(self) -> None:
        """Manual rotation leaves the selected center fixed."""
        matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = add_rotation_about_center(matrix, 90.0, (10.0, 10.0))
        points = np.asarray([[10.0, 10.0, 1.0], [0.0, 10.0, 1.0]])
        actual = result @ points.T
        np.testing.assert_allclose(actual[:, 0], [10.0, 10.0], atol=1e-12)
        np.testing.assert_allclose(actual[:, 1], [10.0, 0.0], atol=1e-12)

    def test_zero_length_pair_is_rejected(self) -> None:
        """Two identical clicks cannot define an alignment direction."""
        with self.assertRaises(ValueError):
            calculate_similarity_alignment([(1, 1), (1, 1)], [(2, 2), (3, 3)])

 
    def test_center_radius_alignment_maps_centers_and_radii(self) -> None:
        """The center/radius transform maps the comparison disk onto the reference."""
        matrix = center_radius_alignment_matrix(
            (120.0, 90.0), 80.0, (300.0, 220.0), 160.0
        )
        center = matrix @ np.asarray([300.0, 220.0, 1.0])
        edge = matrix @ np.asarray([460.0, 220.0, 1.0])
        np.testing.assert_allclose(center, [120.0, 90.0], atol=1e-12)
        np.testing.assert_allclose(edge, [200.0, 90.0], atol=1e-12)

    def test_center_radius_alignment_rejects_invalid_radius(self) -> None:
        """Zero or negative disk radii are not valid scale references."""
        with self.assertRaises(ValueError):
            center_radius_alignment_matrix((0.0, 0.0), 10.0, (1.0, 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
