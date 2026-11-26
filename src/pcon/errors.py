"""OOF の見逃し(FN)・誤検出(FP)を周辺込みで並べた画像を作る。

例: uv run python -m src.pcon.errors --exp base_r18_m64
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from .data import TILE, build_mosaics, crop_context, load_tables
from .train import OUT_DIR, best_f1, sigmoid


def sheet(rows: pd.DataFrame, store, n: int, margin: int = 64, cell: int = 192) -> np.ndarray:
    rows = rows.head(n)
    cols = 6
    nrow = max(1, int(np.ceil(len(rows) / cols)))
    out = np.full((nrow * (cell + 20), cols * cell, 3), 255, np.uint8)
    for k, r in enumerate(rows.itertuples()):
        img = crop_context(store, r.photo, r.gx, r.gy, margin).copy()
        cv2.rectangle(img, (margin, margin), (margin + TILE - 1, margin + TILE - 1), (255, 0, 0), 2)
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        y, x = (k // cols) * (cell + 20), (k % cols) * cell
        out[y : y + cell, x : x + cell] = img
        cv2.putText(out, f"{r.photo}-{r.index} p={r.prob:.2f}", (x + 2, y + cell + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n", type=int, default=36)
    args = ap.parse_args()
    train, _ = load_tables()
    oof = pd.read_csv(OUT_DIR / args.exp / "oof.csv")
    col = "prob" if "prob" in oof else "logit"
    train["prob"] = oof[col].to_numpy() if col == "prob" else sigmoid(oof[col].to_numpy())
    _, thr = best_f1(train["label"].to_numpy(), train["prob"].to_numpy())
    store = build_mosaics()
    fn = train[(train.label == 1) & (train.prob < thr)].sort_values("prob")
    fp = train[(train.label == 0) & (train.prob >= thr)].sort_values("prob", ascending=False)
    print(f"thr={thr:.3f} FN={len(fn)} FP={len(fp)}")
    for name, rows in [("fn", fn), ("fp", fp)]:
        path = OUT_DIR / args.exp / f"errors_{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(sheet(rows, store, args.n), cv2.COLOR_RGB2BGR))
        print(path)
    print(fn[["path", "prob"]].to_string(index=False))
    print(fp[["path", "prob"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
