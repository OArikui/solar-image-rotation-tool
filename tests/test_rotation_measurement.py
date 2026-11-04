"""Tests for three-point rotation and similarity measurements."""

import math
import unittest

import numpy as np

from solar_rotation_tool.rotation_measurement import (
    add_rotation_about_center,
    calculate_rotation_only_angle,
    calculate_similarity_alignment,
    center_radius_alignment_matrix,
    centered_fit_matrix,
)


class RotationMeasurementTests(unittest.TestCase):
    """Verify three-point transforms and image-coordinate angle signs."""

    def test_identity_alignment(self) -> None:
        """An image aligned with itself has zero rotation and unit scale."""
        points = [(10.0, 20.0), (40.0, 20.0), (10.0, 50.0)]
        alignment = calculate_similarity_alignment(points, points)
        np.testing.assert_allclose(
            alignment.matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-12
        )
        self.assertAlmostEqual(alignment.rotation_deg, 0.0)
        self.assertAlmostEqual(alignment.scale, 1.0)

    def test_three_point_similarity_fit_recovers_rotation_scale_and_translation(self) -> None:
        """The least-squares fit recovers a known three-point similarity transform."""
        comparison = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        reference = np.asarray([[5.0, 7.0], [5.0, 27.0], [-15.0, 7.0]])
        alignment = calculate_similarity_alignment(reference, comparison)
        self.assertAlmostEqual(alignment.rotation_deg, 90.0)
        self.assertAlmostEqual(alignment.scale, 2.0)
        np.testing.assert_allclose(
            alignment.transform_points(comparison), reference, atol=1e-12
        )

    def test_clockwise_screen_rotation_is_positive(self) -> None:
        """Image-coordinate angle signs follow x-right/y-down coordinates."""
        comparison = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0)]
        reference = [(2.0, 3.0), (12.0, 3.0), (2.0, -7.0)]
        alignment = calculate_similarity_alignment(reference, comparison)
        self.assertAlmostEqual(alignment.rotation_deg, -90.0)
        np.testing.assert_allclose(
            alignment.transform_points(comparison), reference, atol=1e-12
        )

    def test_rotation_only_ignores_scale_and_translation(self) -> None:
        """Rotation-only fitting estimates angle without adding scale or translation."""
        comparison = np.asarray([[10.0, 20.0], [30.0, 20.0], [10.0, 40.0]])
        reference = np.asarray([[100.0, 200.0], [100.0, 240.0], [60.0, 200.0]])
        angle = calculate_rotation_only_angle(reference, comparison)
        self.assertAlmostEqual(angle, 90.0)

    def test_rotation_only_uses_three_pairs_and_rejects_degenerate_geometry(self) -> None:
        """Three pairs are required and coincident landmarks cannot define an angle."""
        with self.assertRaises(ValueError):
            calculate_rotation_only_angle([(0, 0), (1, 1)], [(0, 0), (1, 1)])
        with self.assertRaises(ValueError):
            calculate_rotation_only_angle(
                [(0, 0), (0, 0), (0, 0)],
                [(1, 1), (1, 1), (1, 1)],
            )

    def test_similarity_fit_rejects_degenerate_points(self) -> None:
        """A collapsed point set cannot define a similarity transform."""
        with self.assertRaises(ValueError):
            calculate_similarity_alignment(
                [(0, 0), (0, 0), (0, 0)],
                [(1, 1), (2, 2), (3, 3)],
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
