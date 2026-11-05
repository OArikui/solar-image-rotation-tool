"""Adapter for the vendored MIN2 v2.4.3 solar-disk detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class DiskGeometry:
    """Detected or manually specified solar-disk geometry in image pixels."""

    center: tuple[float, float]
    radius: float
    method: str


def detect_solar_disk(
    image: np.ndarray,
    *,
    n: int | None = None,
    light_threshold: int | None = None,
    limb_wigth: int | str | None = None,
    iter_cycles: int | None = None,
    config_file: str | Path | None = None,
) -> DiskGeometry:
    """Detect the solar-disk center and radius with MIN2 v2.4.3.

    The bundled YAML configuration is used unless a config_file is supplied.
    Explicit keyword arguments take precedence over values in that file.
    """
    values = np.asarray(image)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("MIN2には空でない2次元グレースケール画像が必要です。")
    if min(values.shape) < 16:
        raise ValueError("MIN2で検出するには画像の縦横がそれぞれ16画素以上必要です。")

    from solar_rotation_tool.vendor.MIN2ver2 import (  # noqa: PLC0415
        MIN2_ignore_sunspots,
        version,
    )

    default_config = Path(__file__).resolve().parent / "vendor" / "min2_config.yaml"
    selected_config = Path(config_file) if config_file is not None else default_config
    (center_x, center_y), radius = MIN2_ignore_sunspots(
        img_inst=values,
        n=n,
        light_threshold=light_threshold,
        limb_wigth=limb_wigth,
        iter_cycles=iter_cycles,
        show=False,
        debug=False,
        show_simple=True,
        config=selected_config,
    )
    center = (float(center_x), float(center_y))
    radius_value = float(radius)
    if not np.all(np.isfinite(center)) or not np.isfinite(radius_value):
        raise ValueError("MIN2が有限な中心座標・半径を返しませんでした。")
    if radius_value <= 0.0:
        raise ValueError(f"MIN2が不正な太陽半径を返しました: {radius_value}")
    return DiskGeometry(center=center, radius=radius_value, method=version)
