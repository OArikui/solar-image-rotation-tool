"""Regression tests for image-canvas zoom and pan coordinate handling."""

import unittest

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

from solar_rotation_tool.image_canvas import ImageCanvas


class ImageCanvasTests(unittest.TestCase):
    """Verify that view transforms do not alter original-image coordinates."""

    _app: QApplication | None = None

    @classmethod
    def setUpClass(cls) -> None:
        """Create one offscreen Qt application for canvas tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        cls._app = app

    def test_image_coordinates_round_trip_after_zoom_and_pan(self) -> None:
        """Zoom and pan still map cursor locations to the source pixel coordinate."""
        canvas = ImageCanvas("Test")
        canvas.resize(620, 420)
        canvas.set_image(np.zeros((180, 240, 3), dtype=np.uint8))
        canvas.zoom_factor = 2.5
        canvas.pan_offset = QPointF(31.0, -27.0)

        source_point = (123.5, 91.25)
        widget_point = canvas._image_to_widget(source_point)
        actual_point = canvas._widget_to_image(widget_point)

        self.assertIsNotNone(actual_point)
        assert actual_point is not None
        np.testing.assert_allclose(np.asarray(actual_point), np.asarray(source_point), atol=1e-10)

    def test_fit_to_view_resets_zoom_and_pan(self) -> None:
        """The fit action restores the default canvas transform."""
        canvas = ImageCanvas("Test")
        canvas.resize(620, 420)
        canvas.set_image(np.zeros((180, 240, 3), dtype=np.uint8))
        canvas.zoom_factor = 3.0
        canvas.pan_offset = QPointF(50.0, -20.0)

        canvas.fit_to_view()

        self.assertEqual(canvas.zoom_factor, 1.0)
        self.assertEqual(canvas.pan_offset, QPointF(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
