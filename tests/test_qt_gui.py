"""GUI-level tests for three-point selection and MIN2 rotation-only fitting."""

import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication

from solar_rotation_tool.min2_adapter import DiskGeometry
from solar_rotation_tool.qt_gui import SolarRotationToolDialog
from solar_rotation_tool.rotation_measurement import center_radius_alignment_matrix

_QT_APP: QApplication | None = None


class SolarRotationDialogTests(unittest.TestCase):
    """Verify point selection and MIN2-plus-rotation GUI behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a Qt application for offscreen dialog tests."""
        global _QT_APP
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            app = QApplication([])
        _QT_APP = app

    def test_collects_three_points_and_restarts_on_the_fourth(self) -> None:
        """Each image stores three ordered points before starting a new set."""
        dialog = SolarRotationToolDialog()
        for index in range(3):
            dialog._add_reference_point(float(index), float(index + 1))
            dialog._add_comparison_point(float(index + 10), float(index + 11))

        self.assertEqual(len(dialog.reference_points), 3)
        self.assertEqual(len(dialog.comparison_points), 3)
        self.assertIn("A: 3/3 点", dialog.measurement_label.text())
        self.assertIn("B: 3/3 点", dialog.measurement_label.text())

        dialog._add_reference_point(99.0, 98.0)
        dialog._add_comparison_point(89.0, 88.0)
        self.assertEqual(dialog.reference_points, [(99.0, 98.0)])
        self.assertEqual(dialog.comparison_points, [(89.0, 88.0)])

    def test_overlay_uses_complementary_cyan_and_red_channels(self) -> None:
        """Identical pixels blend to grey; differences retain their source colors."""
        dialog = SolarRotationToolDialog()
        dialog.reference_rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
        dialog.comparison_rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
        dialog.base_matrix = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        dialog.opacity = 50

        identical = dialog._composite()
        self.assertIsNotNone(identical)
        assert identical is not None
        np.testing.assert_array_equal(identical[3, 3], [50, 50, 50])

        dialog.comparison_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        reference_only = dialog._composite()
        self.assertIsNotNone(reference_only)
        assert reference_only is not None
        np.testing.assert_array_equal(reference_only[3, 3], [0, 50, 50])

        dialog.reference_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        dialog.comparison_rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
        comparison_only = dialog._composite()
        self.assertIsNotNone(comparison_only)
        assert comparison_only is not None
        np.testing.assert_array_equal(comparison_only[3, 3], [50, 0, 0])

    def test_overlay_axes_follow_the_transformed_comparison_up_direction(self) -> None:
        """The B-up arrow follows the linear part of the displayed transform."""
        dialog = SolarRotationToolDialog()
        dialog.reference_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        dialog.comparison_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        dialog.base_matrix = np.asarray(
            [[0.0, -1.0, 100.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
        dialog.alignment_mode = "対応点による相似変換"

        dialog._refresh_overlay()

        axes = dialog.overlay_canvas.orientation_axes
        self.assertIsNotNone(axes)
        assert axes is not None
        _, reference_up, comparison_up = axes
        self.assertEqual(reference_up, (0.0, -1.0))
        np.testing.assert_allclose(comparison_up, [1.0, 0.0], atol=1e-12)
        self.assertTrue(dialog.overlay_canvas.rotation_interaction_enabled)

    def test_min2_mode_changes_only_rotation_about_the_solar_center(self) -> None:
        """The rotation-only option preserves the MIN2 base scale and disk center."""
        dialog = SolarRotationToolDialog()
        dialog.reference_rgb = np.zeros((400, 400, 3), dtype=np.uint8)
        dialog.comparison_rgb = np.zeros((400, 400, 3), dtype=np.uint8)
        reference_disk = DiskGeometry((200.0, 200.0), 100.0, "MIN2 v2.4.1")
        comparison_disk = DiskGeometry((190.0, 205.0), 80.0, "MIN2 v2.4.1")
        dialog.reference_disk = reference_disk
        dialog.comparison_disk = comparison_disk
        base_matrix = center_radius_alignment_matrix(
            reference_disk.center,
            reference_disk.radius,
            comparison_disk.center,
            comparison_disk.radius,
        )
        dialog.base_matrix = base_matrix.copy()
        dialog.scale = 1.25
        dialog.alignment_mode = "MIN2中心・半径補正"
        dialog._update_rotation_only_option()

        self.assertTrue(dialog.rotation_only_checkbox.isEnabled())
        dialog.rotation_only_checkbox.setChecked(True)

        comparison_points = np.asarray(
            [[160.0, 185.0], [230.0, 190.0], [180.0, 250.0]],
            dtype=np.float64,
        )
        angle = np.deg2rad(30.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        comparison_offsets = comparison_points - np.asarray([190.0, 205.0])
        reference_points = comparison_offsets @ rotation.T + np.asarray(
            [200.0, 200.0]
        )
        dialog.comparison_points = [
            (float(x), float(y)) for x, y in comparison_points
        ]
        dialog.reference_points = [
            (float(x), float(y)) for x, y in reference_points
        ]

        dialog._calculate_alignment()

        self.assertEqual(
            dialog.alignment_mode, "MIN2中心・半径補正＋対応点回転のみ"
        )
        self.assertAlmostEqual(dialog.automatic_angle, 30.0, places=8)
        np.testing.assert_allclose(dialog.base_matrix, base_matrix, atol=1e-12)
        self.assertAlmostEqual(dialog.scale, 1.25, places=12)

        matrix = dialog._current_matrix()
        self.assertIsNotNone(matrix)
        assert matrix is not None
        mapped_center = matrix @ np.asarray([190.0, 205.0, 1.0])
        np.testing.assert_allclose(mapped_center, [200.0, 200.0], atol=1e-10)
        self.assertAlmostEqual(
            float(np.linalg.det(matrix[:, :2])),
            float(np.linalg.det(base_matrix[:, :2])),
            places=10,
        )

        dialog.rotation_only_checkbox.setChecked(False)
        self.assertEqual(dialog.alignment_mode, "MIN2中心・半径補正")
        self.assertAlmostEqual(dialog.automatic_angle, 0.0)
        restored_matrix = dialog._current_matrix()
        self.assertIsNotNone(restored_matrix)
        assert restored_matrix is not None
        np.testing.assert_allclose(restored_matrix, base_matrix, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
