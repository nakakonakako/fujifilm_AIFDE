"""提出 v21 の作成。
帯の合成モデル群（CV fold0-2 + 全データ学習3本）と、重点学習モデル群（CV 4実験 + 全データ学習10本）の
テスト予測（logit）をそれぞれ平均して半々で混ぜ、テスト陽性率が学習データの陽性率と同じになる閾値で二値化する。
例: uv run python scripts/make_v21.py --out submission.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pcon.compare import load_test  # noqa: E402
from src.pcon.data import load_tables  # noqa: E402
from src.pcon.train import OUT_DIR, sigmoid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
tr, te = load_tables()

stripe_dir = OUT_DIR / "r18_m64_hard_j0_stripe"
stripe = np.mean([np.load(stripe_dir / f"test_logit_f{k}.npy") for k in range(3)]
                 + [pd.read_csv(OUT_DIR / e / "test_logit.csv").logit.values
                    for e in ["r18_m64_hard_j0_stripe_full", "r18_m64_hard_j0_stripe_full_w"]], 0)
hard = load_test("+".join([
    "r18_m64_hard", "r18_m64_hard_s7", "r18_m64_hard_s11", "r18_m64_hard_j0",
    "r18_m64_hard_full", "r18_m64_hard_full_b", "r18_m64_hard_full_c", "r18_m64_hard_full_d",
    "r18_m64_hard_j0_full", "r18_m64_hard_j0_full_w"]))
p = sigmoid((stripe + hard) / 2)
thr = np.quantile(p, 1 - tr.label.mean())
pred = (p >= thr).astype(int)
pd.DataFrame(dict(path=te.path, pred=pred)).to_csv(a.out, index=False, header=False)
print(f"thr={thr:.4f} test_pos={pred.mean():.2%} -> {a.out}")
