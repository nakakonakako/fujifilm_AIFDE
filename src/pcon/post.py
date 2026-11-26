"""写真内の位置関係を使った後処理（スタッキング）。

各タイルの logit に加えて、近傍タイルの logit や写真内の統計量を特徴量にし、
ロジスティック回帰を写真単位 CV で学習して OOF F1 を評価する。

例: uv run python -m src.pcon.post --exps base_r18_m64 --out post_base
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from .data import GRID_H, GRID_W, load_tables
from .train import OUT_DIR, best_f1, sigmoid


def grid_features(df: pd.DataFrame, logit: np.ndarray) -> np.ndarray:
    """df は photo/gx/gy を持つ。写真ごとに logit をグリッドに並べて特徴量を作る。"""
    feats = np.zeros((len(df), 8), dtype=np.float32)
    df = df.reset_index(drop=True)
    for _, idx in df.groupby("photo").groups.items():
        idx = np.asarray(idx)
        g = np.full((GRID_H + 2, GRID_W + 2), -1e9, dtype=np.float32)
        gx = df.loc[idx, "gx"].to_numpy() + 1
        gy = df.loc[idx, "gy"].to_numpy() + 1
        np.maximum.at(g, (gy, gx), logit[idx])
        g[g <= -1e8] = np.nan
        filled = np.where(np.isnan(g), np.nanmin(logit[idx]), g)
        four = np.stack([filled[gy - 1, gx], filled[gy + 1, gx], filled[gy, gx - 1], filled[gy, gx + 1]], 1)
        diag = np.stack([filled[gy - 1, gx - 1], filled[gy - 1, gx + 1], filled[gy + 1, gx - 1], filled[gy + 1, gx + 1]], 1)
        own = logit[idx]
        p = sigmoid(logit[idx])
        feats[idx, 0] = own
        feats[idx, 1] = four.max(1)
        feats[idx, 2] = np.sort(four, 1)[:, -2]  # 2番目に大きい4近傍
        feats[idx, 3] = diag.max(1)
        feats[idx, 4] = np.log1p(p.sum())  # 写真内の期待陽性数
        feats[idx, 5] = np.sort(own)[-5:].mean()  # 写真内の上位スコア
        feats[idx, 6] = own - np.median(own)
        feats[idx, 7] = np.maximum(own, four.max(1))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True, help="平均する実験名")
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=0.3)
    args = ap.parse_args()

    train, test = load_tables()
    oofs = [pd.read_csv(OUT_DIR / e / "oof.csv") for e in args.exps]
    tes = [pd.read_csv(OUT_DIR / e / "test_logit.csv") for e in args.exps]
    oof = np.mean([o["logit"].to_numpy() for o in oofs], 0)
    te = np.mean([t["logit"].to_numpy() for t in tes], 0)
    fold = oofs[0]["fold"].to_numpy()
    y = train["label"].to_numpy()

    base_f1, base_thr = best_f1(y, sigmoid(oof))
    Xtr = grid_features(train, oof)
    Xte = grid_features(test, te)

    stack = np.zeros(len(train))
    for k in np.unique(fold):
        m = fold == k
        clf = LogisticRegression(C=args.C, max_iter=2000).fit(Xtr[~m], y[~m])
        stack[m] = clf.predict_proba(Xtr[m])[:, 1]
    f, thr = best_f1(y, stack)
    clf = LogisticRegression(C=args.C, max_iter=2000).fit(Xtr, y)
    pte = clf.predict_proba(Xte)[:, 1]

    per_fold = {int(k): float(f1_score(y[fold == k], stack[fold == k] >= thr)) for k in np.unique(fold)}
    summary = dict(
        exps=args.exps,
        base=dict(f1=base_f1, thr=base_thr, ap=average_precision_score(y, oof)),
        stack=dict(f1=f, thr=thr, ap=average_precision_score(y, stack), per_fold=per_fold),
        coef=clf.coef_.round(3).tolist(),
        test_pos_rate=float((pte >= thr).mean()),
    )
    print(json.dumps(summary, indent=2))
    out = OUT_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(dict(path=train["path"], label=y, fold=fold, prob=stack)).to_csv(out / "oof.csv", index=False)
    pd.DataFrame(dict(path=test["path"], pred=(pte >= thr).astype(int))).to_csv(
        out / "submission.csv", index=False, header=False
    )


if __name__ == "__main__":
    main()
