#!/usr/bin/env bash
# 提出 v21 の再現（学習 → 提出CSV作成）。RTX 3060 Ti 1枚で約8時間、RTX 4090 で約3時間。
set -eu
cd "$(dirname "$0")/.."
T="uv run python -m src.pcon.train"
H="--hard-from base_r18_m64 --epochs 30"

uv run python -m src.pcon.data                                  # 元写真の復元キャッシュ
$T --exp base_r18_m64                                           # ベースライン（難例重み付けの元）

# 重点学習（難例重み付け）5-fold CV × 4
$T --exp r18_m64_hard     $H
$T --exp r18_m64_hard_s7  $H --seed 7
$T --exp r18_m64_hard_s11 $H --seed 11
$T --exp r18_m64_hard_j0  $H --jitter 0
# 帯状物体の合成拡張つき 5-fold CV（v21 では fold0-2 のモデルを使用）
$T --exp r18_m64_hard_j0_stripe $H --jitter 0 --stripe 0.3

# 全データ学習（検証なし）
$T --exp r18_m64_hard_full        $H --full 2
$T --exp r18_m64_hard_full_b      $H --full 2 --seed 100
$T --exp r18_m64_hard_full_c      $H --full 2 --seed 400
$T --exp r18_m64_hard_full_d      $H --full 1 --seed 700
$T --exp r18_m64_hard_j0_full     $H --full 2 --seed 500 --jitter 0
$T --exp r18_m64_hard_j0_full_w   $H --full 1 --seed 600 --jitter 0
$T --exp r18_m64_hard_j0_stripe_full   $H --full 2 --seed 800 --jitter 0 --stripe 0.3
$T --exp r18_m64_hard_j0_stripe_full_w $H --full 1 --seed 900 --jitter 0 --stripe 0.3

# 提出CSV
uv run python scripts/make_v21.py --out submission.csv
uv run python -m src.pcon.check_submission submission.csv
