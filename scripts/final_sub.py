"""最終アンサンブルの提出CSVを作る。閾値は OOF の F1 がほぼ平らな範囲（最大-0.005以内）から、
テスト陽性率が学習データの陽性率に最も近いものを選ぶ。
例: uv run python scripts/final_sub.py --cv a b c --full x y --out submissions/20_xxx.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pcon.compare import load_oof, load_test  # noqa: E402
from src.pcon.data import load_tables  # noqa: E402
from src.pcon.train import sigmoid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cv", nargs="+", required=True)
ap.add_argument("--full", nargs="*", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
tr, te = load_tables()
y = tr.label.values
p = sigmoid(load_oof("+".join(a.cv)))
t = sigmoid(load_test("+".join(a.cv + a.full)))
grid = np.linspace(0.3, 0.98, 341)
fs = np.array([f1_score(y, p >= g) for g in grid])
ok = grid[fs >= fs.max() - 0.005]
prior = y.mean()
thr = ok[np.argmin([abs((t >= g).mean() - prior) for g in ok])]
pred = (t >= thr).astype(int)
pd.DataFrame(dict(path=te.path, pred=pred)).to_csv(a.out, index=False, header=False)
print(f"OOF F1max={fs.max():.4f} F1@thr={f1_score(y, p >= thr):.4f} AP={average_precision_score(y, p):.4f} "
      f"thr={thr:.3f} 範囲=[{ok.min():.3f},{ok.max():.3f}] test_pos={pred.mean():.2%} (学習 {prior:.2%}) -> {a.out}")
