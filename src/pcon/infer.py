"""学習済み fold モデルで、入力を拡大した推論（ズームTTA）を行い OOF とテスト予測を保存する。

遠くに写った小さいPコン跡を拡大して見せる狙い。モデルは global pooling なので入力サイズを変えて推論できる。
例: python -m src.pcon.infer --exp r18_m64_hard --zooms 1.25 1.5
出力: outputs/<exp>/oof_z1.25.csv, test_logit_z1.25.csv
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from .data import build_mosaics, load_tables
from .train import OUT_DIR, TileDataset, make_folds, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--zooms", nargs="+", type=float, required=True)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    exp_dir = OUT_DIR / a.exp
    args = SimpleNamespace(**json.loads((exp_dir / "args.json").read_text()))
    mask = getattr(args, "mask", False)
    pad = getattr(args, "pad", "edge")
    train, test = load_tables()
    train["fold"] = make_folds(train, args.n_folds, seed=42)
    store = build_mosaics()
    device = torch.device("cuda")
    kw = dict(num_workers=a.workers, pin_memory=True)

    for z in a.zooms:
        size = int(round(args.size * z / 32) * 32)
        oof = np.full(len(train), np.nan)
        tests = []
        for k in range(args.n_folds):
            model = timm.create_model(args.arch, pretrained=False, num_classes=1, in_chans=4 if mask else 3)
            model.load_state_dict(torch.load(exp_dir / f"model_f{k}.pt", map_location="cpu"))
            model = model.to(device).to(memory_format=torch.channels_last).eval()
            m = (train.fold == k).to_numpy()
            dl = DataLoader(TileDataset(train[m], store, args.margin, size, train=False, mask=mask, pad=pad), batch_size=64, **kw)
            oof[m] = predict(model, dl, device, a.tta)
            dl = DataLoader(TileDataset(test, store, args.margin, size, train=False, mask=mask, pad=pad), batch_size=64, **kw)
            tests.append(predict(model, dl, device, a.tta))
            print(f"z={z} fold{k} done", flush=True)
        pd.DataFrame(dict(path=train["path"], label=train["label"], fold=train["fold"], logit=oof)).to_csv(
            exp_dir / f"oof_z{z}.csv", index=False)
        pd.DataFrame(dict(path=test["path"], logit=np.mean(tests, 0))).to_csv(exp_dir / f"test_logit_z{z}.csv", index=False)


if __name__ == "__main__":
    main()
