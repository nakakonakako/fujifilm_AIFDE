"""実験の比較。写真単位ブートストラップで基準実験との差の信頼区間を出す。

例: uv run python -m src.pcon.compare --ref base_r18_m64 --exps m128_r18_e20 r34_m96_e20
複数実験をまとめて1つにする場合は "a+b" と書く（logit 平均）。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from .data import GRID_W, load_tables
from .train import OUT_DIR, best_f1, sigmoid

HARD_PHOTOS = {"7131": "小さいPコン", "7134": "靴・ボケ"}


def load_oof(name: str) -> np.ndarray:
    return np.mean([pd.read_csv(OUT_DIR / e / "oof.csv")["logit"].to_numpy() for e in name.split("+")], 0)


def load_test(name: str) -> np.ndarray | None:
    try:
        return np.mean([pd.read_csv(OUT_DIR / e / "test_logit.csv")["logit"].to_numpy() for e in name.split("+")], 0)
    except FileNotFoundError:
        return None


def f1_at(y, p, t):
    return f1_score(y, p >= t, zero_division=0)


def describe(train: pd.DataFrame, name: str) -> dict:
    y = train["label"].to_numpy()
    p = sigmoid(load_oof(name))
    f, t = best_f1(y, p)
    grid = np.linspace(0.05, 0.995, 200)
    fs = np.array([f1_at(y, p, g) for g in grid])
    ok = grid[fs >= f - 0.01]
    folds = [f1_at(y[train.fold == k], p[train.fold == k], t) for k in sorted(train.fold.unique())]
    pred = p >= t
    out = dict(
        exp=name, f1=f, thr=t, ap=average_precision_score(y, p), f1_05=f1_at(y, p, 0.5),
        fold_mean=np.mean(folds), fold_std=np.std(folds), thr_width=float(ok.max() - ok.min()) if len(ok) else 0.0,
        FN=int(((~pred) & (y == 1)).sum()), FP=int((pred & (y == 0)).sum()),
        FN_cluster=int(((~pred) & (y == 1) & train.has_pos_nb).sum()),
        FN_isolated=int(((~pred) & (y == 1) & ~train.has_pos_nb).sum()),
    )
    for ph, lab in HARD_PHOTOS.items():
        m = (train.photo == ph).to_numpy()
        out[f"{ph}({lab}) FN/FP"] = f"{int(((~pred) & (y == 1) & m).sum())}/{int((pred & (y == 0) & m).sum())}"
    te = load_test(name)
    out["test_pos%"] = float((sigmoid(te) >= t).mean() * 100) if te is not None else np.nan
    return out


def bootstrap(train: pd.DataFrame, ref: str, exp: str, n: int = 1000, seed: int = 0) -> dict:
    y = train["label"].to_numpy()
    pr, pe = sigmoid(load_oof(ref)), sigmoid(load_oof(exp))
    _, tr_ = best_f1(y, pr)
    _, te_ = best_f1(y, pe)
    photos = train.photo.unique()
    idx = {ph: np.flatnonzero(train.photo.to_numpy() == ph) for ph in photos}
    rng = np.random.default_rng(seed)
    df1, dap = [], []
    for _ in range(n):
        sel = np.concatenate([idx[ph] for ph in rng.choice(photos, len(photos))])
        if y[sel].sum() == 0:
            continue
        df1.append(f1_at(y[sel], pe[sel], te_) - f1_at(y[sel], pr[sel], tr_))
        dap.append(average_precision_score(y[sel], pe[sel]) - average_precision_score(y[sel], pr[sel]))
    q = lambda a: np.percentile(a, [2.5, 50, 97.5]).round(4).tolist()
    return dict(exp=exp, dF1_CI=q(df1), P_dF1_gt0=float(np.mean(np.array(df1) > 0)), dAP_CI=q(dap),
                P_dAP_gt0=float(np.mean(np.array(dap) > 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--exps", nargs="*", default=[])
    args = ap.parse_args()
    train, _ = load_tables()
    train["fold"] = pd.read_csv(OUT_DIR / args.ref.split("+")[0] / "oof.csv")["fold"].to_numpy()
    # 4近傍に陽性があるか（穴が境界をまたぐケース）
    pos = set(zip(train.photo[train.label == 1], train["index"][train.label == 1]))
    def nb(r):
        cand = [r["index"] - GRID_W, r["index"] + GRID_W]
        if r.gx > 0:
            cand.append(r["index"] - 1)
        if r.gx < GRID_W - 1:
            cand.append(r["index"] + 1)
        return any((r.photo, c) in pos for c in cand)
    train["has_pos_nb"] = train.apply(nb, axis=1)

    pd.set_option("display.width", 250)
    rows = [describe(train, e) for e in [args.ref, *args.exps]]
    print(pd.DataFrame(rows).set_index("exp").round(4).T.to_string())
    if args.exps:
        print()
        print(pd.DataFrame([bootstrap(train, args.ref, e) for e in args.exps]).set_index("exp").to_string())


if __name__ == "__main__":
    main()
