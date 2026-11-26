"""提出CSVの形式チェック。例: uv run python -m src.pcon.check_submission outputs/xxx/submission.csv"""

from __future__ import annotations

import sys

import pandas as pd

from .data import ROOT


def main(path: str) -> None:
    sub = pd.read_csv(path, header=None, names=["path", "pred"], dtype={"path": str})
    test = [l.strip() for l in (ROOT / "test.txt").read_text().splitlines() if l.strip()]
    errors = []
    if len(sub) != len(test):
        errors.append(f"行数 {len(sub)} != test.txt {len(test)}")
    if sub["path"].tolist() != test:
        errors.append("画像パスの並びが test.txt と一致しない")
    if not sub["pred"].isin([0, 1]).all():
        errors.append("0/1 以外の値がある")
    if sub["path"].iloc[0] == "path" or str(sub["pred"].iloc[0]) == "pred":
        errors.append("ヘッダー行がある")
    with open(path, "rb") as f:
        raw = f.read()
    size_mb = len(raw) / 1e6
    if size_mb > 10:
        errors.append(f"サイズ {size_mb:.2f}MB > 10MB")
    print(f"rows={len(sub)} pos={int(sub['pred'].sum())} ({sub['pred'].mean():.3%}) size={size_mb:.2f}MB")
    print("OK" if not errors else "NG: " + " / ".join(errors))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main(sys.argv[1])
