"""OOF に基づくアンサンブルの貪欲選択（重複選択あり = 重み付き平均）と提出CSV作成。

例:
  uv run python -m src.pcon.ensemble --cands base_r18_m64 r18_m64_hard r18_m64_hard_s7 \
      --full r18_m64_hard=r18_m64_hard_full --out submissions/05_ens.csv
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import load_tables
from .train import OUT_DIR, best_f1, sigmoid


def score(y: np.ndarray, logit: np.ndarray) -> tuple[float, float, float]:
    f, t = best_f1(y, sigmoid(logit))
    return f, average_precision_score(y, logit), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True, help="候補のCV実験名")
    ap.add_argument("--full", nargs="*", default=[], help="CV実験=全データ実験（テスト予測にのみ加える）")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--metric", choices=["f1", "ap"], default="ap", help="貪欲選択の基準（F1は陽性が少なく不安定なのでAP推奨）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train, test = load_tables()
    y = train["label"].to_numpy()
    oof, te = {}, {}
    for c in args.cands:
        if not (OUT_DIR / c / "oof.csv").exists():
            print(f"skip {c}: 未完了")
            continue
        o = pd.read_csv(OUT_DIR / c / "oof.csv")["logit"].to_numpy()
        if np.isnan(o).any():
            print(f"skip {c}: OOF が全foldそろっていない")
            continue
        oof[c] = o
        te[c] = pd.read_csv(OUT_DIR / c / "test_logit.csv")["logit"].to_numpy()
    for pair in args.full:
        cv, full = pair.split("=")
        if cv in te:
            tf = pd.read_csv(OUT_DIR / full / "test_logit.csv")["logit"].to_numpy()
            te[cv] = (te[cv] + tf) / 2  # CV 5本平均と全データ学習の平均を半々
            print(f"{cv}: テスト予測に {full} を追加")

    print("単体:")
    for c in oof:
        f, a, t = score(y, oof[c])
        print(f"  {c:28s} f1={f:.4f} ap={a:.4f} thr={t:.3f}")

    key = 0 if args.metric == "f1" else 1
    chosen: list[str] = []
    best = (-1.0, -1.0, 0.5)
    for _ in range(args.max_steps):
        trial = []
        for c in oof:
            s = score(y, np.mean([oof[k] for k in chosen + [c]], 0))
            trial.append((s[key], s, c))
        trial.sort(key=lambda r: r[0], reverse=True)
        top, s, c = trial[0]
        if chosen and top <= best[key] + 1e-4:
            break
        chosen.append(c)
        best = s
        print(f"+ {c:28s} -> f1={s[0]:.4f} ap={s[1]:.4f} thr={s[2]:.3f}")

    w = Counter(chosen)
    print("選択:", dict(w))
    f, a, t = best
    te_logit = np.mean([te[k] for k in chosen], 0)
    pred = (sigmoid(te_logit) >= t).astype(int)
    print(f"OOF f1={f:.4f} ap={a:.4f} thr={t:.4f} test_pos={pred.mean():.2%}")
    if args.out:
        pd.DataFrame(dict(path=test["path"], pred=pred)).to_csv(args.out, index=False, header=False)
        print("->", args.out)


if __name__ == "__main__":
    main()
