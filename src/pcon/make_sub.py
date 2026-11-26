"""複数実験の logit 平均で提出CSVを作る。閾値はアンサンブルOOFで最適化。

例: uv run python -m src.pcon.make_sub --exps base_r18_m64 r34_m96_e20 --out submissions/03_xxx.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .compare import load_oof, load_test
from .data import load_tables
from .train import best_f1, sigmoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True, help="OOFと閾値決めに使う実験（logit平均）")
    ap.add_argument("--test-extra", nargs="*", default=[], help="テスト予測だけに加える実験（全データ学習など）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train, test = load_tables()
    y = train["label"].to_numpy()
    name = "+".join(args.exps)
    oof = load_oof(name)
    f, t = best_f1(y, sigmoid(oof))
    te = load_test("+".join(args.exps + args.test_extra))
    pred = (sigmoid(te) >= t).astype(int)
    pd.DataFrame(dict(path=test["path"], pred=pred)).to_csv(args.out, index=False, header=False)
    print(f"oof_f1={f:.4f} ap={average_precision_score(y, oof):.4f} thr={t:.4f} test_pos={pred.mean():.2%} -> {args.out}")


if __name__ == "__main__":
    main()
