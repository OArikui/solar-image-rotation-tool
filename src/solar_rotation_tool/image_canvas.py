"""Interactive image canvas with cursor-centred zoom, pan and disk editing."""

from __future__ import annotations

from math import hypot

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PyQt6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    """Display an image with landmark selection and optional disk-circle editing."""

    point_clicked = pyqtSignal(float, float)
    disk_changed = pyqtSignal(float, float, float)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.image: QImage | None = None
        self._image_buffer: np.ndarray | None = None
        self.image_width = 0
        self.image_height = 0
        self.points: list[tuple[float, float]] = []
        self.disk_center: tuple[float, float] | None = None
        self.disk_radius: float | None = None
        self.manual_disk_edit_enabled = False
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self._panning = False
        self._last_pan_position: QPointF | None = None
        self._disk_drag_mode: str | None = None
        self.setMinimumSize(320, 280)
        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray | None, *, reset_view: bool = True) -> None:
        """Replace the displayed image."""
        if image is None:
            self.image = None
            self._image_buffer = None
            self.image_width = 0
            self.image_height = 0
        else:
            self._image_buffer = np.ascontiguousarray(image, dtype=np.uint8).copy()
            height, width = self._image_buffer.shape[:2]
            image_bytes = self._image_buffer.tobytes()
            self.image = QImage(
                image_bytes,
                width,
                height,
                int(self._image_buffer.strides[0]),
                QImage.Format.Format_RGB888,
            ).copy()
            self.image_height = height
            self.image_width = width
        self.points.clear()
        if reset_view:
            self.fit_to_view()
        else:
            self.update()

    def set_disk_circle(
        self,
        center: tuple[float, float] | None,
        radius: float | None,
    ) -> None:
        """Set or clear the detected/manual solar-disk circle."""
        self.disk_center = center
        self.disk_radius = radius
        self.update()

    def set_manual_disk_edit_enabled(self, enabled: bool) -> None:
        """Enable dragging the disk center or radius handle."""
        self.manual_disk_edit_enabled = enabled
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.CrossCursor
        )
        self.update()

    def fit_to_view(self) -> None:
        """Reset zoom and pan so the complete image is visible."""
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self.update()

    def _fit_scale(self) -> float:
        if self.image is None or self.image_width <= 0 or self.image_height <= 0:
            return 1.0
        return min(
            max(1, self.width() - 24) / self.image_width,
            max(1, self.height() - 44) / self.image_height,
        )

    def _image_rect(self) -> QRectF:
        """Return the transformed image rectangle in widget coordinates."""
        if self.image is None:
            return QRectF()
        scale = self._fit_scale() * self.zoom_factor
        width = self.image_width * scale
        height = self.image_height * scale
        return QRectF(
            (self.width() - width) / 2.0 + self.pan_offset.x(),
            (self.height() - height) / 2.0 + 10.0 + self.pan_offset.y(),
            width,
            height,
        )

    def _widget_to_image(
        self,
        position: QPointF,
        *,
        require_inside: bool = True,
    ) -> tuple[float, float] | None:
        """Convert a widget coordinate to original-image pixel coordinates."""
        if self.image is None:
            return None
        rect = self._image_rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return None
        if require_inside and not rect.contains(position):
            return None
        scale = rect.width() / self.image_width
        x = (position.x() - rect.left()) / scale
        y = (position.y() - rect.top()) / scale
        if not require_inside:
            x = min(max(x, 0.0), float(self.image_width - 1))
            y = min(max(y, 0.0), float(self.image_height - 1))
        return float(x), float(y)

    def _image_to_widget(self, point: tuple[float, float]) -> QPointF:
        rect = self._image_rect()
        scale = rect.width() / self.image_width
        return QPointF(rect.left() + point[0] * scale, rect.top() + point[1] * scale)

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802
        """Draw the image, landmarks and optional solar-disk guide."""
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(27, 31, 38))
        painter.setPen(QColor(235, 239, 244))
        painter.drawText(10, 20, self.title)
        if self.image is None:
            painter.setPen(QColor(155, 163, 175))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "画像を読み込んでください",
            )
            return

        rect = self._image_rect()
        painter.drawImage(rect, self.image)
        painter.setPen(QPen(QColor(80, 88, 100), 1))
        painter.drawRect(rect)

        for index, (x, y) in enumerate(self.points, start=1):
            marker = self._image_to_widget((x, y))
            painter.setPen(QPen(QColor(255, 220, 70), 2))
            painter.drawEllipse(marker, 7, 7)
            painter.setPen(QColor(255, 245, 180))
            painter.drawText(marker + QPointF(9, -8), str(index))

        if self.disk_center is not None and self.disk_radius is not None:
            center = self._image_to_widget(self.disk_center)
            scale = rect.width() / self.image_width
            radius = self.disk_radius * scale
            handle = QPointF(center.x() + radius, center.y())
            painter.setPen(QPen(QColor(60, 220, 255), 2))
            painter.drawEllipse(center, radius, radius)
            painter.drawLine(
                QPointF(center.x() - 9, center.y()),
                QPointF(center.x() + 9, center.y()),
            )
            painter.drawLine(
                QPointF(center.x(), center.y() - 9),
                QPointF(center.x(), center.y() + 9),
            )
            painter.setBrush(QColor(60, 220, 255))
            handle_size = 5 if self.manual_disk_edit_enabled else 3
            painter.drawEllipse(handle, handle_size, handle_size)
            painter.setBrush(Qt.BrushStyle.NoBrush)

    def wheelEvent(self, event: QWheelEvent | None) -> None:  # noqa: N802
        """Zoom around the cursor while keeping the pixel beneath it stationary."""
        if event is None or self.image is None:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        cursor_position = event.position()
        image_position = self._widget_to_image(cursor_position, require_inside=False)
        if image_position is None:
            event.ignore()
            return

        factor = 1.2 ** (delta / 120.0)
        new_zoom = min(max(self.zoom_factor * factor, 0.25), 24.0)
        new_scale = self._fit_scale() * new_zoom
        image_x, image_y = image_position
        self.zoom_factor = new_zoom
        base_left = (self.width() - self.image_width * new_scale) / 2.0
        base_top = (self.height() - self.image_height * new_scale) / 2.0 + 10.0
        self.pan_offset = QPointF(
            cursor_position.x() - image_x * new_scale - base_left,
            cursor_position.y() - image_y * new_scale - base_top,
        )
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        """Start panning, editing a disk circle, or selecting a landmark."""
        if event is None or self.image is None:
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._last_pan_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        if not self.manual_disk_edit_enabled:
            self.point_clicked.emit(*image_point)
            event.accept()
            return

        self._start_disk_edit(event.position(), image_point)
        event.accept()

    def _start_disk_edit(
        self,
        widget_position: QPointF,
        image_point: tuple[float, float],
    ) -> None:
        """Choose whether a drag moves the center, changes radius or creates a circle."""
        if self.disk_center is not None and self.disk_radius is not None:
            center_widget = self._image_to_widget(self.disk_center)
            radius_widget = QPointF(
                center_widget.x()
                + self.disk_radius * self._image_rect().width() / self.image_width,
                center_widget.y(),
            )
            if hypot(
                widget_position.x() - radius_widget.x(),
                widget_position.y() - radius_widget.y(),
            ) <= 18.0:
                self._disk_drag_mode = "radius"
                return
            if hypot(
                widget_position.x() - center_widget.x(),
                widget_position.y() - center_widget.y(),
            ) <= 18.0:
                self._disk_drag_mode = "center"
                return

        self.disk_center = image_point
        self.disk_radius = 1.0
        self._disk_drag_mode = "new_radius"
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        """Update pan or the active manual disk-circle drag."""
        if event is None:
            return
        if self._panning and self._last_pan_position is not None:
            position = event.position()
            delta = position - self._last_pan_position
            self.pan_offset += delta
            self._last_pan_position = position
            self.update()
            return

        if self._disk_drag_mode is None or self.disk_center is None:
            return
        image_point = self._widget_to_image(event.position(), require_inside=False)
        if image_point is None:
            return
        if self._disk_drag_mode == "center":
            self.disk_center = image_point
        else:
            self.disk_radius = max(
                1.0,
                hypot(
                    image_point[0] - self.disk_center[0],
                    image_point[1] - self.disk_center[1],
                ),
            )
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        """Finish panning or publish the manually edited disk circle."""
        if event is None:
            return
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._panning = False
            self._last_pan_position = None
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if self.manual_disk_edit_enabled
                else Qt.CursorShape.CrossCursor
            )
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._disk_drag_mode is not None:
            self._disk_drag_mode = None
            if self.disk_center is not None and self.disk_radius is not None:
                self.disk_radius = max(1.0, self.disk_radius)
                self.disk_changed.emit(
                    self.disk_center[0],
                    self.disk_center[1],
                    self.disk_radius,
                )
            self.update()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        """Fit the image to the canvas on a left-button double click."""
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_view()
            event.accept()
