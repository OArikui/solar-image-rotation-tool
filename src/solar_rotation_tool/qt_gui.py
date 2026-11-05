"""PyQt GUI for solar-image registration and relative rotation measurement."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from solar_rotation_tool.image_canvas import ImageCanvas
from solar_rotation_tool.image_display import to_rgb
from solar_rotation_tool.image_io import load_image
from solar_rotation_tool.min2_adapter import DiskGeometry, detect_solar_disk
from solar_rotation_tool.rotation_measurement import (
    add_rotation_about_center,
    calculate_rotation_only_angle,
    calculate_similarity_alignment,
    center_radius_alignment_matrix,
    centered_fit_matrix,
)

IMAGE_FILTER = "Solar images (*.tif *.tiff *.fits *.fit *.fts);;All files (*)"


def _rgb_image(image: np.ndarray) -> np.ndarray:
    """Convert a 2D scientific image into an independent RGB preview."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("空でない2次元のグレースケール画像を指定してください。")
    return to_rgb(image)


class SolarRotationToolDialog(QDialog):
    """Interactive two-image registration, disk calibration and overlay tool."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("太陽画像の回転角測定")
        self.resize(1450, 1050)
        self.reference_path: Path | None = None
        self.comparison_path: Path | None = None
        self.reference: np.ndarray | None = None
        self.comparison: np.ndarray | None = None
        self.reference_rgb: np.ndarray | None = None
        self.comparison_rgb: np.ndarray | None = None
        self.reference_points: list[tuple[float, float]] = []
        self.comparison_points: list[tuple[float, float]] = []
        self.reference_disk: DiskGeometry | None = None
        self.comparison_disk: DiskGeometry | None = None
        self.base_matrix: np.ndarray | None = None
        self.alignment_mode = "initial"
        self.automatic_angle = 0.0
        self.scale = 1.0
        self.manual_angle = 0.0
        self._overlay_drag_base_angle = 0.0
        self.opacity = 50

        root = QVBoxLayout(self)
        file_row = QHBoxLayout()
        self.reference_button = QPushButton("画像 A を開く")
        self.comparison_button = QPushButton("画像 B を開く")
        self.reference_name = QLabel("画像 A: 未選択")
        self.comparison_name = QLabel("画像 B: 未選択")
        self.reference_button.clicked.connect(self._open_reference)
        self.comparison_button.clicked.connect(self._open_comparison)
        file_row.addWidget(self.reference_button)
        file_row.addWidget(self.reference_name, 1)
        file_row.addSpacing(12)
        file_row.addWidget(self.comparison_button)
        file_row.addWidget(self.comparison_name, 1)
        root.addLayout(file_row)

        root.addWidget(
            QLabel(
                "Zoom: ホイール（カーソル位置中心） / Pan: 中ボタンまたは右ボタンをドラッグ"
                " / ダブルクリック: 表示範囲をリセット"
                " / 重ね合わせ画面: 左ドラッグでBを回転"
            )
        )

        canvases = QHBoxLayout()
        self.reference_canvas = ImageCanvas("A: 基準画像")
        self.comparison_canvas = ImageCanvas("B: 比較画像")
        self.reference_canvas.point_clicked.connect(self._add_reference_point)
        self.comparison_canvas.point_clicked.connect(self._add_comparison_point)
        self.reference_canvas.disk_changed.connect(self._on_reference_disk_changed)
        self.comparison_canvas.disk_changed.connect(self._on_comparison_disk_changed)
        canvases.addWidget(self.reference_canvas, 1)
        canvases.addWidget(self.comparison_canvas, 1)
        root.addLayout(canvases, 3)

        disk_row = QHBoxLayout()
        self.detect_disk_button = QPushButton("MIN2 v2.4.3で中心・半径を自動検出")
        self.manual_disk_checkbox = QCheckBox("中心・半径をマウスで手動調整")
        self.reset_view_button = QPushButton("全画像の表示をリセット")
        self.detect_disk_button.clicked.connect(self._detect_disk_geometry)
        self.manual_disk_checkbox.toggled.connect(self._set_manual_disk_edit)
        self.reset_view_button.clicked.connect(self._fit_all_canvases)
        disk_row.addWidget(self.detect_disk_button)
        disk_row.addWidget(self.manual_disk_checkbox)
        disk_row.addWidget(self.reset_view_button)
        root.addLayout(disk_row)

        self.disk_status_label = QLabel(
            "中心・半径: 未検出（MIN2自動検出、または手動調整を使用できます）"
        )
        self.disk_status_label.setWordWrap(True)
        root.addWidget(self.disk_status_label)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("B の透過率"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(self.opacity)
        self.opacity_label = QLabel("50%")
        self.opacity_slider.valueChanged.connect(self._set_opacity)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_label)
        root.addLayout(opacity_row)

        adjustment_row = QHBoxLayout()
        adjustment_row.addWidget(QLabel("追加回転（画像座標・時計回り +）"))
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-180.0, 180.0)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.valueChanged.connect(self._set_manual_rotation)
        adjustment_row.addWidget(self.rotation_spin)
        self.rotation_only_checkbox = QCheckBox(
            "MIN2と併用（対応点から回転のみ補正）"
        )
        self.rotation_only_checkbox.setEnabled(False)
        self.rotation_only_checkbox.setToolTip(
            "MIN2で求めた中心・半径・倍率・平行移動は維持し、"
            "対応点3組から回転角だけを算出します。"
        )
        self.rotation_only_checkbox.toggled.connect(self._set_rotation_only_mode)
        adjustment_row.addWidget(self.rotation_only_checkbox)
        self.calculate_button = QPushButton("対応点3組から回転・倍率を補正")
        self.clear_button = QPushButton("対応点をクリア")
        self.reset_button = QPushButton("位置合わせを初期化")
        self.save_button = QPushButton("重ね合わせ画像・測定値を保存")
        self.calculate_button.clicked.connect(self._calculate_alignment)
        self.clear_button.clicked.connect(self._clear_points)
        self.reset_button.clicked.connect(self._reset_alignment)
        self.save_button.clicked.connect(self._save_results)
        adjustment_row.addWidget(self.calculate_button)
        adjustment_row.addWidget(self.clear_button)
        adjustment_row.addWidget(self.reset_button)
        adjustment_row.addWidget(self.save_button)
        root.addLayout(adjustment_row)

        self.overlay_canvas = ImageCanvas(
            "補色重ね合わせ（A: シアン / B: レッド）  A上:シアン矢印 / B上:赤矢印"
        )
        self.overlay_canvas.rotation_drag_started.connect(
            self._begin_overlay_rotation_drag
        )
        self.overlay_canvas.rotation_dragged.connect(
            self._on_overlay_rotation_dragged
        )
        self.overlay_canvas.rotation_drag_finished.connect(
            self._finish_overlay_rotation_drag
        )
        self.overlay_canvas.setToolTip(
            "左ドラッグでB画像を回転します。シアン矢印はA、赤矢印は変換後のBの上方向です。"
        )
        self.measurement_label = QLabel(
            "画像 A・B を読み込み、MIN2による中心・半径補正、または対応点を指定してください。"
        )
        self.measurement_label.setWordWrap(True)
        root.addWidget(self.overlay_canvas, 2)
        root.addWidget(self.measurement_label)

    def _load(self, path: str, reference: bool) -> None:
        """Load and validate one TIFF/FITS grayscale image."""
        try:
            image = load_image(path)
            rgb = _rgb_image(image)
        except Exception as error:
            QMessageBox.critical(self, "画像を読み込めません", f"{path}\n\n{error}")
            return

        if reference:
            self.reference_path, self.reference, self.reference_rgb = Path(path), image, rgb
            self.reference_name.setText(str(path))
            self.reference_canvas.set_image(rgb)
            self.reference_canvas.set_disk_circle(None, None)
            self.reference_disk = None
        else:
            self.comparison_path, self.comparison, self.comparison_rgb = Path(path), image, rgb
            self.comparison_name.setText(str(path))
            self.comparison_canvas.set_image(rgb)
            self.comparison_canvas.set_disk_circle(None, None)
            self.comparison_disk = None

        self._clear_points()
        self._reset_alignment(show_message=False)
        self._update_disk_status()

    def _open_reference(self) -> None:
        """Open the reference-image file picker."""
        path, _ = QFileDialog.getOpenFileName(
            self, "基準画像 A を選択", "", IMAGE_FILTER
        )
        if path:
            self._load(path, True)

    def _open_comparison(self) -> None:
        """Open the comparison-image file picker."""
        path, _ = QFileDialog.getOpenFileName(
            self, "比較画像 B を選択", "", IMAGE_FILTER
        )
        if path:
            self._load(path, False)

    def _add_reference_point(self, x: float, y: float) -> None:
        """Append a reference-image landmark, keeping at most two."""
        if len(self.reference_points) >= 3:
            self.reference_points.clear()
        self.reference_points.append((x, y))
        self.reference_canvas.points = self.reference_points.copy()
        self.reference_canvas.update()
        self._update_measurement()

    def _add_comparison_point(self, x: float, y: float) -> None:
        """Append a comparison-image landmark, keeping at most two."""
        if len(self.comparison_points) >= 3:
            self.comparison_points.clear()
        self.comparison_points.append((x, y))
        self.comparison_canvas.points = self.comparison_points.copy()
        self.comparison_canvas.update()
        self._update_measurement()

    def _update_rotation_only_option(self) -> None:
        """Enable MIN2-plus-landmark rotation mode only when both disk circles exist."""
        has_disk_alignment = (
            self.reference_disk is not None
            and self.comparison_disk is not None
            and self.alignment_mode
            in {
                "MIN2中心・半径補正",
                "手動中心・半径補正",
                "MIN2中心・半径補正＋対応点回転のみ",
                "手動中心・半径補正＋対応点回転のみ",
            }
        )
        if not has_disk_alignment and self.rotation_only_checkbox.isChecked():
            self.rotation_only_checkbox.blockSignals(True)
            self.rotation_only_checkbox.setChecked(False)
            self.rotation_only_checkbox.blockSignals(False)
        self.rotation_only_checkbox.setEnabled(has_disk_alignment)
        self.calculate_button.setText(
            "対応点3組から回転のみ補正"
            if self.rotation_only_checkbox.isChecked()
            else "対応点3組から回転・倍率を補正"
        )

    def _set_rotation_only_mode(self, enabled: bool) -> None:
        """Switch between full similarity fitting and MIN2 plus rotation-only fitting."""
        if enabled and (
            self.reference_disk is None or self.comparison_disk is None
        ):
            self.rotation_only_checkbox.setChecked(False)
            QMessageBox.warning(
                self,
                "MIN2補正が必要です",
                "先にMIN2で画像A/Bの太陽円を検出するか、両方の円を手動設定してください。",
            )
            return
        if not enabled and "対応点回転のみ" in self.alignment_mode:
            method = (
                "手動中心・半径補正"
                if self.reference_disk is not None
                and (
                    self.reference_disk.method == "manual"
                    or (
                        self.comparison_disk is not None
                        and self.comparison_disk.method == "manual"
                    )
                )
                else "MIN2中心・半径補正"
            )
            self._apply_disk_alignment(method)
            return
        self.calculate_button.setText(
            "対応点3組から回転のみ補正"
            if enabled
            else "対応点3組から回転・倍率を補正"
        )
        self._update_measurement()

    def _detect_disk_geometry(self) -> None:
        """Run MIN2 v2.4.3 independently on the two source images."""
        if self.reference is None or self.comparison is None:
            QMessageBox.warning(self, "画像未選択", "画像 A と画像 B を読み込んでください。")
            return

        failures: list[str] = []
        self.reference_disk = None
        self.comparison_disk = None
        self.reference_canvas.set_disk_circle(None, None)
        self.comparison_canvas.set_disk_circle(None, None)

        try:
            self.reference_disk = detect_solar_disk(self.reference)
            self.reference_canvas.set_disk_circle(
                self.reference_disk.center, self.reference_disk.radius
            )
        except Exception as error:
            failures.append(f"画像 A: {error}")

        try:
            self.comparison_disk = detect_solar_disk(self.comparison)
            self.comparison_canvas.set_disk_circle(
                self.comparison_disk.center, self.comparison_disk.radius
            )
        except Exception as error:
            failures.append(f"画像 B: {error}")

        self._update_disk_status()
        if self.reference_disk is not None and self.comparison_disk is not None:
            self._apply_disk_alignment("MIN2中心・半径補正")
        else:
            self._update_rotation_only_option()
            self._update_measurement()

        if failures:
            QMessageBox.warning(
                self,
                "MIN2の検出結果",
                "一部の画像で太陽円を検出できませんでした。\n"
                "手動調整モードで中心と半径を指定できます。\n\n"
                + "\n".join(failures),
            )

    def _set_manual_disk_edit(self, enabled: bool) -> None:
        """Toggle mouse editing for the disk circles in both source canvases."""
        self.reference_canvas.set_manual_disk_edit_enabled(enabled)
        self.comparison_canvas.set_manual_disk_edit_enabled(enabled)
        if enabled:
            self.disk_status_label.setText(
                "手動調整: 円の中心マーカーをドラッグすると中心を移動します。"
                "右端のハンドルをドラッグすると半径を変更します。"
                "円の外側からドラッグを始めると新しい円を作成します。"
                "中/右ボタンはPan、ホイールはZoomです。"
            )
        else:
            self._update_disk_status()

    def _on_reference_disk_changed(self, x: float, y: float, radius: float) -> None:
        """Store a manually adjusted reference-image disk circle."""
        self.reference_disk = DiskGeometry((x, y), radius, "manual")
        self._update_disk_status()
        self._apply_manual_disk_alignment()

    def _on_comparison_disk_changed(self, x: float, y: float, radius: float) -> None:
        """Store a manually adjusted comparison-image disk circle."""
        self.comparison_disk = DiskGeometry((x, y), radius, "manual")
        self._update_disk_status()
        self._apply_manual_disk_alignment()

    def _apply_manual_disk_alignment(self) -> None:
        """Apply center/radius alignment when both manual circles are available."""
        if self.reference_disk is not None and self.comparison_disk is not None:
            self._apply_disk_alignment("手動中心・半径補正")

    def _apply_disk_alignment(self, method: str) -> None:
        """Map B's detected/manual disk center and radius onto image A."""
        if self.reference_disk is None or self.comparison_disk is None:
            return
        try:
            self.base_matrix = center_radius_alignment_matrix(
                self.reference_disk.center,
                self.reference_disk.radius,
                self.comparison_disk.center,
                self.comparison_disk.radius,
            )
        except ValueError as error:
            QMessageBox.warning(self, "中心・半径補正に失敗しました", str(error))
            return

        self.alignment_mode = method
        self.automatic_angle = 0.0
        self.scale = self.reference_disk.radius / self.comparison_disk.radius
        self._reset_manual_rotation()
        self._update_rotation_only_option()
        self._refresh_overlay()
        self._update_measurement()

    def _update_disk_status(self) -> None:
        """Show each image's current disk center and radius."""
        def describe(label: str, disk: DiskGeometry | None) -> str:
            if disk is None:
                return f"{label}: 未設定"
            return (
                f"{label}: 中心=({disk.center[0]:.2f}, {disk.center[1]:.2f}) px, "
                f"半径={disk.radius:.2f} px [{disk.method}]"
            )

        self.disk_status_label.setText(
            describe("A", self.reference_disk) + "　 |　 "
            + describe("B", self.comparison_disk)
        )

    def _calculate_alignment(self) -> None:
        """Calculate a three-point fit, or only rotation when paired with MIN2."""
        if self.reference_rgb is None or self.comparison_rgb is None:
            QMessageBox.warning(self, "画像未選択", "画像 A と画像 B を読み込んでください。")
            return

        if self.rotation_only_checkbox.isChecked():
            if (
                self.reference_disk is None
                or self.comparison_disk is None
                or self.base_matrix is None
            ):
                QMessageBox.warning(
                    self, "MIN2補正が必要です", "先に画像A/Bの中心・半径補正を行ってください。"
                )
                return
            try:
                angle = calculate_rotation_only_angle(
                    self.reference_points, self.comparison_points
                )
            except ValueError as error:
                QMessageBox.warning(self, "対応点が不足しています", str(error))
                return

            self.automatic_angle = angle
            prefix = (
                "手動中心・半径補正"
                if self.reference_disk.method == "manual"
                or self.comparison_disk.method == "manual"
                else "MIN2中心・半径補正"
            )
            self.alignment_mode = f"{prefix}＋対応点回転のみ"
        else:
            try:
                result = calculate_similarity_alignment(
                    self.reference_points, self.comparison_points
                )
            except ValueError as error:
                QMessageBox.warning(self, "対応点が不足しています", str(error))
                return

            self.base_matrix = result.matrix
            self.alignment_mode = "対応点による相似変換"
            self.automatic_angle = result.rotation_deg
            self.scale = result.scale

        self._reset_manual_rotation()
        self._refresh_overlay()
        self._update_measurement()

    def _clear_points(self) -> None:
        """Clear both landmark selections without changing the current alignment."""
        self.reference_points.clear()
        self.comparison_points.clear()
        self.reference_canvas.points = []
        self.comparison_canvas.points = []
        self.reference_canvas.update()
        self.comparison_canvas.update()
        self._update_measurement()

    def _reset_manual_rotation(self) -> None:
        """Reset the additional rotation control without emitting another update."""
        self.manual_angle = 0.0
        self.rotation_spin.blockSignals(True)
        self.rotation_spin.setValue(0.0)
        self.rotation_spin.blockSignals(False)

    def _reset_alignment(self, show_message: bool = True) -> None:
        """Return to a centered, aspect-ratio-preserving initial placement."""
        self.alignment_mode = "initial"
        self.automatic_angle = 0.0
        self._reset_manual_rotation()
        self._update_rotation_only_option()
        if self.reference_rgb is not None and self.comparison_rgb is not None:
            self.base_matrix = centered_fit_matrix(
                self.reference_rgb.shape[:2], self.comparison_rgb.shape[:2]
            )
            self.scale = float(math.sqrt(abs(np.linalg.det(self.base_matrix[:, :2]))))
        else:
            self.base_matrix = None
            self.scale = 1.0
        self._refresh_overlay()
        self._update_measurement()
        if show_message:
            self.measurement_label.setText(
                self.measurement_label.text() + "\n位置合わせを初期配置に戻しました。"
            )

    def _fit_all_canvases(self) -> None:
        """Reset zoom and pan on A, B and the overlay view."""
        self.reference_canvas.fit_to_view()
        self.comparison_canvas.fit_to_view()
        self.overlay_canvas.fit_to_view()

    def _begin_overlay_rotation_drag(self) -> None:
        """Remember the current manual angle before an overlay drag begins."""
        self._overlay_drag_base_angle = self.manual_angle

    def _on_overlay_rotation_dragged(self, delta_degrees: float) -> None:
        """Apply the overlay drag delta through the shared numeric rotation control."""
        self.rotation_spin.setValue(self._overlay_drag_base_angle + delta_degrees)

    def _finish_overlay_rotation_drag(self) -> None:
        """Keep the final drag angle as the base for the next drag gesture."""
        self._overlay_drag_base_angle = self.manual_angle

    def _set_manual_rotation(self, value: float) -> None:
        """Apply an additional image-coordinate rotation to image B."""
        self.manual_angle = float(value)
        self._refresh_overlay()
        self._update_measurement()

    def _set_opacity(self, value: int) -> None:
        """Update the overlay transparency."""
        self.opacity = int(value)
        self.opacity_label.setText(f"{value}%")
        self._refresh_overlay()

    def _rotation_center(self) -> tuple[float, float]:
        """Return the solar-disk center for disk-aligned overlays, else image center."""
        if self.reference_rgb is None:
            return (0.0, 0.0)
        height, width = self.reference_rgb.shape[:2]
        disk_alignment_modes = {
            "MIN2中心・半径補正",
            "手動中心・半径補正",
            "MIN2中心・半径補正＋対応点回転のみ",
            "手動中心・半径補正＋対応点回転のみ",
        }
        has_disk_alignment = (
            self.alignment_mode in disk_alignment_modes
            and self.reference_disk is not None
            and self.comparison_disk is not None
            and self.base_matrix is not None
        )
        if has_disk_alignment and self.reference_disk is not None:
            return self.reference_disk.center
        return (width / 2.0, height / 2.0)

    def _current_matrix(self) -> np.ndarray | None:
        """Return the full comparison-to-reference image transform."""
        if self.base_matrix is None or self.reference_rgb is None:
            return None
        rotation_only_mode = "対応点回転のみ" in self.alignment_mode
        rotation_center = (
            self.reference_disk.center
            if rotation_only_mode and self.reference_disk is not None
            else self._rotation_center()
        )
        applied_angle = (
            self.automatic_angle + self.manual_angle
            if rotation_only_mode
            else self.manual_angle
        )
        return add_rotation_about_center(
            self.base_matrix, applied_angle, rotation_center
        )

    def _warped_comparison(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Warp comparison RGB pixels and a validity mask into reference space."""
        if self.reference_rgb is None or self.comparison_rgb is None:
            return None
        matrix = self._current_matrix()
        if matrix is None:
            return None
        height, width = self.reference_rgb.shape[:2]
        warped = cv2.warpAffine(
            self.comparison_rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        source_mask = np.full(self.comparison_rgb.shape[:2], 255, dtype=np.uint8)
        valid = cv2.warpAffine(
            source_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.ascontiguousarray(warped), np.ascontiguousarray(valid)

    def _composite(self) -> np.ndarray | None:
        """Blend complementary cyan/red images while preserving B's footprint."""
        if self.reference_rgb is None:
            return None

        reference_intensity = self.reference_rgb[..., 0]
        reference_tinted = np.zeros_like(self.reference_rgb)
        reference_tinted[..., 1] = reference_intensity
        reference_tinted[..., 2] = reference_intensity

        warped_result = self._warped_comparison()
        if warped_result is None:
            return reference_tinted

        warped, valid = warped_result
        comparison_tinted = np.zeros_like(warped)
        comparison_tinted[..., 0] = warped[..., 0]
        alpha = (valid.astype(np.float32) / 255.0) * (self.opacity / 100.0)
        blend = reference_tinted.astype(np.float32) * (1.0 - alpha[..., None])
        blend += comparison_tinted.astype(np.float32) * alpha[..., None]
        return np.clip(blend, 0, 255).astype(np.uint8)

    def _refresh_overlay(self) -> None:
        """Refresh the complementary overlay and its orientation/rotation guides."""
        self.overlay_canvas.set_image(self._composite(), reset_view=False)
        matrix = self._current_matrix()
        if matrix is None or self.reference_rgb is None:
            self.overlay_canvas.set_orientation_axes(None)
            self.overlay_canvas.set_rotation_interaction_enabled(False)
            return

        pivot = self._rotation_center()
        reference_up = (0.0, -1.0)
        comparison_up_array = matrix[:, :2] @ np.asarray([0.0, -1.0])
        comparison_up = (
            float(comparison_up_array[0]),
            float(comparison_up_array[1]),
        )
        self.overlay_canvas.set_orientation_axes(
            pivot, reference_up, comparison_up
        )
        self.overlay_canvas.set_rotation_interaction_enabled(True, pivot)

    def _update_measurement(self) -> None:
        """Update landmark status and derived rotation values."""
        point_status = (
            f"A: {len(self.reference_points)}/3 点　 B: {len(self.comparison_points)}/3 点"
        )
        total_correction = self.automatic_angle + self.manual_angle
        if self.alignment_mode == "MIN2中心・半径補正＋対応点回転のみ":
            alignment_status = (
                f"MIN2中心・半径補正 + 対応点から回転のみ補正: "
                f"{self.automatic_angle:+.3f}° / 倍率: {self.scale:.6f}"
            )
        elif self.alignment_mode == "手動中心・半径補正＋対応点回転のみ":
            alignment_status = (
                f"手動中心・半径補正 + 対応点から回転のみ補正: "
                f"{self.automatic_angle:+.3f}° / 倍率: {self.scale:.6f}"
            )
        elif self.alignment_mode == "MIN2中心・半径補正":
            rotation_hint = (
                "（対応点から回転のみ補正を選択中）"
                if self.rotation_only_checkbox.isChecked()
                else "（回転は未補正）"
            )
            alignment_status = (
                f"MIN2 v2.4.3による中心・半径補正{rotation_hint} "
                f"/ 倍率: {self.scale:.6f}"
            )
        elif self.alignment_mode == "手動中心・半径補正":
            rotation_hint = (
                "（対応点から回転のみ補正を選択中）"
                if self.rotation_only_checkbox.isChecked()
                else "（回転は未補正）"
            )
            alignment_status = (
                f"手動中心・半径補正{rotation_hint} / 倍率: {self.scale:.6f}"
            )
        elif self.alignment_mode == "対応点による相似変換":
            alignment_status = (
                f"対応点による自動補正角: {self.automatic_angle:+.3f}° "
                f"/ 倍率: {self.scale:.6f}"
            )
        else:
            alignment_status = "位置合わせ: 中央・アスペクト比を保持して初期配置"

        self.measurement_label.setText(
            f"{point_status}\n{alignment_status}\n"
            f"B に適用した総補正角: {total_correction:+.3f}°\n"
            f"A に対する B の相対回転角: {-total_correction:+.3f}°"
        )

    def _save_results(self) -> None:
        """Save the blended PNG and a JSON record of the measurement."""
        composite = self._composite()
        matrix = self._current_matrix()
        if composite is None or matrix is None or self.comparison_rgb is None:
            QMessageBox.warning(
                self, "保存できません", "画像を2枚読み込み、位置合わせを行ってください。"
            )
            return
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "重ね合わせ画像を保存",
            "sunspot_overlay.png",
            "PNG image (*.png)",
        )
        if not path_text:
            return
        output_path = Path(path_text)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        if not cv2.imwrite(
            str(output_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        ):
            QMessageBox.critical(
                self, "保存失敗", f"PNG を保存できませんでした: {output_path}"
            )
            return

        total_correction = self.automatic_angle + self.manual_angle
        record: dict[str, Any] = {
            "reference_image": str(self.reference_path) if self.reference_path else None,
            "comparison_image": str(self.comparison_path) if self.comparison_path else None,
            "coordinate_convention": (
                "pixel x right, y down; positive rotation is clockwise on screen"
            ),
            "alignment_method": self.alignment_mode,
            "landmark_fit_mode": (
                "rotation_only"
                if "対応点回転のみ" in self.alignment_mode
                else "similarity"
                if self.alignment_mode == "対応点による相似変換"
                else None
            ),
            "automatic_rotation_pivot_px": (
                list(self.reference_disk.center)
                if "対応点回転のみ" in self.alignment_mode
                and self.reference_disk is not None
                else None
            ),
            "reference_disk": asdict(self.reference_disk) if self.reference_disk else None,
            "comparison_disk": asdict(self.comparison_disk) if self.comparison_disk else None,
            "reference_points_px": self.reference_points,
            "comparison_points_px": self.comparison_points,
            "automatic_correction_angle_deg": self.automatic_angle,
            "manual_additional_angle_deg": self.manual_angle,
            "total_correction_angle_deg": total_correction,
            "comparison_relative_rotation_deg": -total_correction,
            "comparison_to_reference_scale": self.scale,
            "comparison_to_reference_matrix": matrix.tolist(),
            "overlay_opacity_percent": self.opacity,
            "overlay_colormap": {
                "reference": "cyan",
                "comparison": "red",
                "method": "complementary_color_alpha_blend",
            },
            "overlay_axes_origin_px": list(self._rotation_center()),
            "reference_image_up_vector_px": [0.0, -1.0],
            "comparison_image_up_vector_in_reference": (
                matrix[:, :2] @ np.asarray([0.0, -1.0])
            ).tolist(),
            "overlay_image": str(output_path),
        }
        json_path = output_path.with_suffix(".json")
        try:
            json_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            QMessageBox.warning(self, "JSON 保存失敗", str(error))
            return
        QMessageBox.information(
            self, "保存しました", f"画像: {output_path}\n測定値: {json_path}"
        )


def run_rotation_tool() -> int:
    """Launch the standalone solar-image rotation measurement GUI."""
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    dialog = SolarRotationToolDialog()
    dialog.show()
    return application.exec()
