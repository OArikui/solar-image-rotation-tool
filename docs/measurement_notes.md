# 測定仕様と注意事項

## 位置合わせの計算

各画像で指定した2点の線分について、方向角・距離を計算します。画像Bの線分を画像Aの線分へ一致させる角度と距離比を求め、最初の対応点が一致するように平行移動を加えます。これにより、画像Bから画像Aへの2×3変換行列を得ます。

この変換は回転・一様倍率・平行移動のみです。せん断、非一様倍率、レンズ歪み、鏡像反転は扱いません。

## 北方向の推定に利用する場合

このツールが得るのは2画像間の相対角度です。相対角度だけから各画像の絶対的な太陽北方向を一意に決めることはできません。北方向が既知の基準画像を用意し、その基準方向に相対角度を適用して比較画像の方向を求めてください。鏡像反転の有無は別途確認してください。

## JSONの主な項目

| 項目 | 内容 |
|---|---|
| `reference_image` | 基準画像Aのパス |
| `comparison_image` | 比較画像Bのパス |
| `reference_points_px` | 画像Aで選択した2点のピクセル座標 |
| `comparison_points_px` | 画像Bで選択した2点のピクセル座標 |
| `automatic_correction_angle_deg` | 対応点から計算した自動補正角 |
| `manual_additional_angle_deg` | GUIで追加した微調整角 |
| `total_correction_angle_deg` | Bに適用した総補正角 |
| `comparison_relative_rotation_deg` | Aを基準とするBの相対回転角 |
| `comparison_to_reference_scale` | BからAへの一様倍率 |
| `comparison_to_reference_matrix` | BからAへの2×3変換行列 |
| `overlay_opacity_percent` | 保存時のB画像の透過率 |

座標は各画像の左上画素を基準とする0始まりのピクセル座標です。Xは右向き、Yは下向きです。
