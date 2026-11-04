"""Geometry helpers for aligning two solar images from matching landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SimilarityAlignment:
    """A similarity transform mapping comparison-image pixels to reference pixels."""

    matrix: NDArray[np.float64]
    rotation_deg: float
    scale: float

    def transform_points(
        self,
        points: NDArray[np.floating[Any]] | list[tuple[float, float]],
    ) -> NDArray[np.float64]:
        """Transform an N-by-2 array of pixel coordinates."""
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("座標は N×2 の配列にしてください。")
        homogeneous = np.column_stack((values, np.ones(len(values))))
        return homogeneous @ self.matrix.T


def _validate_three_point_sets(
    reference_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
    comparison_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return three ordered correspondence pairs."""
    reference = np.asarray(reference_points, dtype=np.float64)
    comparison = np.asarray(comparison_points, dtype=np.float64)
    if reference.shape != (3, 2) or comparison.shape != (3, 2):
        raise ValueError("各画像で対応点を3点ずつ指定してください。")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(comparison)):
        raise ValueError("対応点の座標は有限値にしてください。")
    return reference, comparison


def _rotation_covariance(
    reference: NDArray[np.float64],
    comparison: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return rotation covariance terms and comparison spread."""
    reference_centered = reference - np.mean(reference, axis=0)
    comparison_centered = comparison - np.mean(comparison, axis=0)
    dot_sum = float(np.sum(comparison_centered * reference_centered))
    cross_sum = float(
        np.sum(
            comparison_centered[:, 0] * reference_centered[:, 1]
            - comparison_centered[:, 1] * reference_centered[:, 0]
        )
    )
    comparison_spread = float(np.sum(comparison_centered**2))
    return dot_sum, cross_sum, comparison_spread


def calculate_similarity_alignment(
    reference_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
    comparison_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
) -> SimilarityAlignment:
    """Fit rotation, uniform scale and translation to three matching point pairs.

    The three point pairs are fitted with a least-squares similarity transform.
    The matrix maps comparison-image coordinates to reference-image coordinates.
    With image coordinates (x right, y down), positive angles are clockwise.
    """
    reference, comparison = _validate_three_point_sets(
        reference_points, comparison_points
    )
    dot_sum, cross_sum, comparison_spread = _rotation_covariance(
        reference, comparison
    )
    covariance_length = math.hypot(dot_sum, cross_sum)
    if comparison_spread <= 1e-9 or covariance_length <= 1e-9:
        raise ValueError("対応点の配置から相似変換を計算できません。")

    angle = math.atan2(cross_sum, dot_sum)
    scale = covariance_length / comparison_spread
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    linear = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    reference_center = np.mean(reference, axis=0)
    comparison_center = np.mean(comparison, axis=0)
    translation = reference_center - linear @ comparison_center
    matrix = np.column_stack((linear, translation))
    rotation_deg = (math.degrees(angle) + 180.0) % 360.0 - 180.0
    return SimilarityAlignment(matrix=matrix, rotation_deg=rotation_deg, scale=scale)


def calculate_rotation_only_angle(
    reference_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
    comparison_points: NDArray[np.floating[Any]] | list[tuple[float, float]],
) -> float:
    """Estimate only the rotation angle from three corresponding point pairs.

    Translation and scale are intentionally not calculated or modified. The
    least-squares angle is measured around the centroid of each point set and
    should be applied around the reference solar-disk center after MIN2 fitting.
    """
    reference, comparison = _validate_three_point_sets(
        reference_points, comparison_points
    )
    dot_sum, cross_sum, comparison_spread = _rotation_covariance(
        reference, comparison
    )
    covariance_length = math.hypot(dot_sum, cross_sum)
    if comparison_spread <= 1e-9 or covariance_length <= 1e-9:
        raise ValueError("対応点の配置から回転角を計算できません。")
    angle = math.atan2(cross_sum, dot_sum)
    return (math.degrees(angle) + 180.0) % 360.0 - 180.0

def centered_fit_matrix(
    reference_shape: tuple[int, int],
    comparison_shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Create a non-distorting initial fit that centers the comparison image."""
    reference_height, reference_width = reference_shape
    comparison_height, comparison_width = comparison_shape
    if min(reference_height, reference_width, comparison_height, comparison_width) <= 0:
        raise ValueError("画像サイズは正の値にしてください。")
    scale = min(
        reference_width / comparison_width,
        reference_height / comparison_height,
    )
    offset_x = (reference_width - comparison_width * scale) / 2.0
    offset_y = (reference_height - comparison_height * scale) / 2.0
    return np.asarray(
        [[scale, 0.0, offset_x], [0.0, scale, offset_y]],
        dtype=np.float64,
    )


def add_rotation_about_center(
    matrix: NDArray[np.floating[Any]],
    angle_deg: float,
    center: tuple[float, float],
) -> NDArray[np.float64]:
    """Compose an image-coordinate rotation around a point after an existing transform."""
    source_matrix = np.asarray(matrix, dtype=np.float64)
    if source_matrix.shape != (2, 3):
        raise ValueError("変換行列は 2×3 にしてください。")
    if not math.isfinite(angle_deg):
        raise ValueError("回転角は有限値にしてください。")
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    center_x, center_y = center
    rotation = np.asarray(
        [
            [cosine, -sine, center_x - cosine * center_x + sine * center_y],
            [sine, cosine, center_y - sine * center_x - cosine * center_y],
        ],
        dtype=np.float64,
    )
    homogeneous = np.vstack((source_matrix, [0.0, 0.0, 1.0]))
    rotation_homogeneous = np.vstack((rotation, [0.0, 0.0, 1.0]))
    return (rotation_homogeneous @ homogeneous)[:2]


def center_radius_alignment_matrix(
    reference_center: tuple[float, float],
    reference_radius: float,
    comparison_center: tuple[float, float],
    comparison_radius: float,
) -> NDArray[np.float64]:
    """Map a comparison solar-disk center/radius onto a reference disk.

    The resulting transform applies uniform scaling and translation only.
    It deliberately does not estimate rotation.
    """
    reference = np.asarray(reference_center, dtype=np.float64)
    comparison = np.asarray(comparison_center, dtype=np.float64)
    if reference.shape != (2,) or comparison.shape != (2,):
        raise ValueError("中心座標はそれぞれ (x, y) の2要素にしてください。")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(comparison)):
        raise ValueError("中心座標は有限値にしてください。")
    if not math.isfinite(reference_radius) or not math.isfinite(comparison_radius):
        raise ValueError("半径は有限値にしてください。")
    if reference_radius <= 0.0 or comparison_radius <= 0.0:
        raise ValueError("半径は正の値にしてください。")

    scale = reference_radius / comparison_radius
    translation = reference - scale * comparison
    return np.asarray(
        [[scale, 0.0, translation[0]], [0.0, scale, translation[1]]],
        dtype=np.float64,
    )
