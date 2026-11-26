"""写真単位 StratifiedGroupKFold で学習し、OOF とテスト予測を保存する。

例: uv run python -m src.pcon.train --exp base_r18 --arch resnet18d --margin 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import ROOT, MosaicStore, build_mosaics, crop_context, load_tables

OUT_DIR = ROOT / "outputs"
cv2.setNumThreads(0)


def make_folds(train: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    fold = np.full(len(train), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sgkf.split(train, train["label"], train["photo"])):
        fold[va] = k
    return fold


class TileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, store: MosaicStore, margin: int, size: int, train: bool, jitter: int = 24,
                 scale_max: float = 1.0, mask: bool = False, pad: str = "edge", stripe: float = 0.0):
        self.photo = df["photo"].to_numpy()
        self.gx = df["gx"].to_numpy()
        self.gy = df["gy"].to_numpy()
        self.label = df["label"].to_numpy().astype(np.float32)
        self.store = store
        self.margin = margin
        self.size = size
        self.train = train
        self.jitter = jitter if train else 0
        # 撮影距離の違い（Pコン跡の見かけの大きさ）に対する拡大縮小。縮小側（視野を広げる）のみ
        self.scale_max = scale_max if train else 1.0
        # 対象タイルの範囲を示す4ch目（平行移動・拡大縮小しても境界の位置をモデルに伝える）
        self.mask = mask
        self.pad = pad
        # 帯状の暗い物体（型枠跡・配管など）を合成してPコン跡と区別させる拡張の確率
        self.stripe = stripe if train else 0.0

    def __len__(self) -> int:
        return len(self.photo)

    def __getitem__(self, i: int):
        m, j = self.margin, self.jitter
        base = 256 + 2 * m
        extra = int(np.ceil(base * (self.scale_max - 1) / 2))
        img = crop_context(self.store, self.photo[i], int(self.gx[i]), int(self.gy[i]), m + j + extra, self.pad)
        if self.mask:
            off = m + j + extra
            mk = np.zeros(img.shape[:2], np.uint8)
            mk[off : off + 256, off : off + 256] = 255
            img = np.concatenate([img, mk[..., None]], axis=2)
        if self.train:
            # 視野サイズ L を [base, base*scale_max] から選び、中心付近を平行移動の揺らぎ付きで切り出す
            L = int(np.random.uniform(base, base * self.scale_max)) if self.scale_max > 1 else base
            c = img.shape[0] // 2
            dy, dx = np.random.randint(-j, j + 1, size=2)
            y0, x0 = c - L // 2 + dy, c - L // 2 + dx
            img = img[y0 : y0 + L, x0 : x0 + L]
            rgb = img[..., :3]
            if self.stripe > 0 and np.random.rand() < self.stripe:
                rgb = self._add_stripe(rgb)
            img = np.concatenate([self._photometric(rgb), img[..., 3:]], axis=2)
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            if img.ndim == 2:
                img = img[..., None]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # uint8 CHW
        return x, self.label[i]

    @staticmethod
    def _add_stripe(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        canvas = np.zeros((h, w), np.uint8)
        ang = np.random.uniform(0, np.pi)
        cx, cy = np.random.uniform(0, w), np.random.uniform(0, h)
        d = np.array([np.cos(ang), np.sin(ang)]) * 2 * max(h, w)
        p1 = (int(cx - d[0]), int(cy - d[1]))
        p2 = (int(cx + d[0]), int(cy + d[1]))
        cv2.line(canvas, p1, p2, 255, int(np.random.uniform(25, 90)))
        a = cv2.GaussianBlur(canvas, (0, 0), 2).astype(np.float32)[..., None] / 255 * np.random.uniform(0.4, 0.7)
        out = img.astype(np.float32) * (1 - a) + img.astype(np.float32) * np.random.uniform(0.35, 0.6) * a
        return out.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _photometric(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        # 明るさ・コントラスト・色味の揺らぎ
        a = np.random.uniform(0.7, 1.3)
        b = np.random.uniform(-25, 25)
        mean = img.mean()
        img = (img - mean) * np.random.uniform(0.7, 1.3) + mean * a + b
        img = img * np.random.uniform(0.92, 1.08, size=3)
        if np.random.rand() < 0.3:
            gray = img.mean(2, keepdims=True)
            img = gray + (img - gray) * np.random.uniform(0.0, 1.0)
        if np.random.rand() < 0.2:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if np.random.rand() < 0.2:
            img = img + np.random.normal(0, np.random.uniform(2, 8), img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """8通りの反転・回転（k=0..7）。x は (B,C,H,W)。"""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def to_input(x: torch.Tensor, device) -> torch.Tensor:
    x = x.to(device, non_blocking=True).float().div_(255.0)
    x = (x - 0.5) / 0.25
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device, tta: int) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        x = to_input(x, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = sum(model(dihedral(x, k)).float() for k in range(tta)) / tta
        out.append(logit.squeeze(1).cpu().numpy())
    return np.concatenate(out)


def best_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best = (0.0, 0.5)
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 500))):
        f = f1_score(y, p >= t)
        if f > best[0]:
            best = (float(f), float(t))
    return best


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_pseudo(test: pd.DataFrame, exps: list[str], lo: float, hi: float) -> pd.DataFrame:
    """指定実験のテスト予測（平均logit）から確信度の高いものだけ疑似ラベル化する。"""
    logit = np.mean([pd.read_csv(OUT_DIR / e / "test_logit.csv")["logit"].to_numpy() for e in exps], 0)
    p = sigmoid(logit)
    keep = (p >= hi) | (p <= lo)
    ps = test[keep].copy()
    ps["label"] = (p[keep] >= hi).astype(int)
    print(f"pseudo: {len(ps)}/{len(test)} 件（陽性 {int(ps.label.sum())}）", flush=True)
    return ps


def train_fold(args, fold: int, train: pd.DataFrame, test: pd.DataFrame, store: MosaicStore, exp_dir: Path,
               pseudo: pd.DataFrame | None = None):
    device = torch.device("cuda")
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    # fold < 0 は全データ学習（検証なし）
    tr = train[train.fold != fold].reset_index(drop=True)
    va = train[train.fold == fold].reset_index(drop=True)
    full = len(va) == 0
    n_real = len(tr)
    if pseudo is not None:
        tr = pd.concat([tr, pseudo], ignore_index=True)

    ds_tr = TileDataset(tr, store, args.margin, args.size, train=True, jitter=args.jitter, scale_max=args.scale_max, mask=args.mask, pad=args.pad, stripe=args.stripe)
    pos = tr["label"].to_numpy() == 1
    w = np.where(pos, args.pos_frac / pos.mean(), (1 - args.pos_frac) / (1 - pos.mean()))
    if args.hard_from:
        # 前の実験の OOF で間違えやすかった実データを多めにサンプリング（疑似ラベル行は対象外）
        prev = sigmoid(np.mean([pd.read_csv(OUT_DIR / e / "oof.csv")["logit"].to_numpy() for e in args.hard_from], 0))
        p_prev = np.zeros(len(tr))
        p_prev[:n_real] = prev[tr["index_in_train"].to_numpy()[:n_real].astype(int)]
        err = np.where(pos, 1 - p_prev, p_prev)
        err[n_real:] = 0
        w = w * (1 + args.hard_alpha * err)
    # 1エポックのサンプル数は実データ分に固定（疑似ラベル追加で学習時間を増やさない）
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=n_real, replacement=True)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, sampler=sampler, drop_last=True, **kw)
    dl_va = None if full else DataLoader(
        TileDataset(va, store, args.margin, args.size, train=False, mask=args.mask, pad=args.pad), batch_size=args.bs * 2, **kw)

    model = timm.create_model(args.arch, pretrained=False, num_classes=1, drop_rate=0.2, drop_path_rate=0.1,
                              in_chans=4 if args.mask else 3)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=args.ema, use_warmup=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl_tr)
    warm = len(dl_tr) * 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    crit = nn.BCEWithLogitsLoss()

    log = []
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl_tr:
            x = to_input(x, device)
            x = dihedral(x, int(np.random.randint(8)))
            y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x).float().squeeze(1), y * (1 - args.smooth) + 0.5 * args.smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            step += 1
            ema.update(model, step=step)
            tot += loss.item()
        if full:
            print(f"[full{fold}] ep{ep:02d} loss {tot / len(dl_tr):.4f} ({time.time()-t0:.0f}s)", flush=True)
        elif ep % args.eval_every == args.eval_every - 1 or ep == args.epochs - 1:
            p = sigmoid(predict(ema.module, dl_va, device, tta=1))
            yv = va["label"].to_numpy()
            f, t = best_f1(yv, p)
            rec = dict(ep=ep, loss=tot / len(dl_tr), ap=average_precision_score(yv, p), f1=f, thr=t)
            log.append(rec)
            print(f"[fold{fold}] ep{ep:02d} loss {rec['loss']:.4f} ap {rec['ap']:.4f} f1 {f:.4f}@{t:.3f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save(ema.module.state_dict(), exp_dir / f"model_f{fold}.pt")
    oof = np.zeros(0) if full else predict(ema.module, dl_va, device, tta=args.tta)
    dl_te = DataLoader(TileDataset(test, store, args.margin, args.size, train=False, mask=args.mask, pad=args.pad), batch_size=args.bs * 2, **kw)
    te = predict(ema.module, dl_te, device, tta=args.tta)
    return va.index_in_train.to_numpy(), oof, te, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arch", default="resnet18d")
    ap.add_argument("--margin", type=int, default=64)
    ap.add_argument("--size", type=int, default=384, help="モデル入力サイズ（切り出し後にリサイズ）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--pos-frac", type=float, default=0.15)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--folds", type=str, default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--hard-from", nargs="*", default=[], help="難例の重み付けに使うOOFの実験名")
    ap.add_argument("--hard-alpha", type=float, default=5.0)
    ap.add_argument("--full", type=int, default=0, help="全データ学習するモデル本数（0ならCV）")
    ap.add_argument("--stripe", type=float, default=0.0, help="帯状の暗い物体を合成する確率")
    ap.add_argument("--pad", choices=["edge", "mean"], default="edge", help="写真外の埋め方")
    ap.add_argument("--mask", action="store_true", help="対象タイル範囲のマスクを4ch目として入力")
    ap.add_argument("--jitter", type=int, default=24)
    ap.add_argument("--scale-max", type=float, default=1.0, help="学習時に視野を最大何倍まで広げるか")
    ap.add_argument("--pseudo", nargs="*", default=[], help="疑似ラベルに使うテスト予測の実験名")
    ap.add_argument("--pseudo-lo", type=float, default=0.02)
    ap.add_argument("--pseudo-hi", type=float, default=0.9)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    exp_dir = OUT_DIR / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    train, test = load_tables()
    train["fold"] = make_folds(train, args.n_folds, seed=42)
    train["index_in_train"] = np.arange(len(train))
    store = build_mosaics()

    pseudo = load_pseudo(test, args.pseudo, args.pseudo_lo, args.pseudo_hi) if args.pseudo else None
    if args.full:
        # 全データ学習: seed を変えて複数本。閾値は対応するCV実験のOOFから決める
        tests = []
        for k in range(args.full):
            args.seed += 1000 * (k + 1)
            _, _, t, _ = train_fold(args, -1 - k, train, test, store, exp_dir, pseudo)
            tests.append(t)
            np.save(exp_dir / f"test_logit_full{k}.npy", t)
        pd.DataFrame(dict(path=test["path"], logit=np.mean(tests, 0))).to_csv(exp_dir / "test_logit.csv", index=False)
        return
    folds = range(args.n_folds) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    oof = np.full(len(train), np.nan)
    tests, logs = [], {}
    for k in folds:
        idx, o, t, log = train_fold(args, k, train, test, store, exp_dir, pseudo)
        oof[idx] = o
        tests.append(t)
        logs[k] = log
        np.save(exp_dir / f"test_logit_f{k}.npy", t)

    m = ~np.isnan(oof)
    y = train["label"].to_numpy()[m]
    p = sigmoid(oof[m])
    f, thr = best_f1(y, p)
    summary = dict(
        oof_f1=f, thr=thr, oof_ap=average_precision_score(y, p), oof_auc=roc_auc_score(y, p),
        f1_at_05=float(f1_score(y, p >= 0.5)), folds=list(folds),
    )
    print(json.dumps(summary, indent=2))
    (exp_dir / "summary.json").write_text(json.dumps(dict(summary=summary, logs=logs), indent=2))
    pd.DataFrame(dict(path=train["path"], label=train["label"], fold=train["fold"], logit=oof)).to_csv(
        exp_dir / "oof.csv", index=False
    )
    te_logit = np.mean(tests, axis=0)
    pd.DataFrame(dict(path=test["path"], logit=te_logit)).to_csv(exp_dir / "test_logit.csv", index=False)
    pd.DataFrame(dict(path=test["path"], pred=(sigmoid(te_logit) >= thr).astype(int))).to_csv(
        exp_dir / "submission.csv", index=False, header=False
    )


if __name__ == "__main__":
    main()
