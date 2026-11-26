"""データ読み込みと、タイルから元写真（18x14タイル）を復元するキャッシュ処理。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "DATA_Maguire_20180517_ALL"
CACHE_DIR = ROOT / "data" / "cache"

TILE = 256
GRID_W = 18  # 1枚の写真の横方向タイル数
GRID_H = 14  # 縦方向タイル数（18x14=252）

_NAME_RE = re.compile(r"(\d+)-(\d+)(?:_\d+)?\.jpg$", re.IGNORECASE)


def parse_path(path: str) -> tuple[str, int]:
    """`W/CW/7107-28_2.jpg` -> ("7107", 28)。`_2` は同じ位置の別名。"""
    m = _NAME_RE.search(path)
    if m is None:
        raise ValueError(f"unexpected file name: {path}")
    return m.group(1), int(m.group(2))


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "train_label.csv", header=None, names=["path", "label"])
    test = pd.read_csv(ROOT / "test.txt", header=None, names=["path"])
    test["label"] = -1
    for df in (train, test):
        parsed = df["path"].map(parse_path)
        df["photo"] = parsed.map(lambda x: x[0])
        df["index"] = parsed.map(lambda x: x[1])
        df["gx"] = (df["index"] - 1) % GRID_W
        df["gy"] = (df["index"] - 1) // GRID_W
    return train, test


@dataclass
class MosaicStore:
    """全写真を復元したuint8配列（memmap）。欠損タイルは valid=False。"""

    images: np.ndarray  # (N, H, W, 3)
    valid: np.ndarray  # (N, GRID_H, GRID_W) bool
    photo_to_slot: dict[str, int]


def build_mosaics(force: bool = False) -> MosaicStore:
    train, test = load_tables()
    allp = pd.concat([train, test], ignore_index=True)
    photos = sorted(allp["photo"].unique())
    slot = {p: i for i, p in enumerate(photos)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    img_path = CACHE_DIR / "mosaics.u8"
    valid_path = CACHE_DIR / "mosaics_valid.npy"
    shape = (len(photos), GRID_H * TILE, GRID_W * TILE, 3)

    if force or not img_path.exists() or not valid_path.exists():
        images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8, shape=shape)
        valid = np.zeros((len(photos), GRID_H, GRID_W), dtype=bool)
        # 同じ位置に複数ファイルがある場合（7074-105 と 105_2 は同一画像）は後勝ち
        for row in allp.itertuples():
            s = slot[row.photo]
            tile = np.asarray(Image.open(DATA_DIR / row.path).convert("RGB"))
            y0, x0 = row.gy * TILE, row.gx * TILE
            images[s, y0 : y0 + TILE, x0 : x0 + TILE] = tile
            valid[s, row.gy, row.gx] = True
        images.flush()
        del images
        np.save(valid_path, valid)

    images = np.load(img_path, mmap_mode="r")
    valid = np.load(valid_path)
    return MosaicStore(images=images, valid=valid, photo_to_slot=slot)


def crop_context(store: MosaicStore, photo: str, gx: int, gy: int, margin: int) -> np.ndarray:
    """対象タイルの周囲 margin px を含めた切り出し。写真外は端の延長、欠損タイルは平均色で埋める。"""
    s = store.photo_to_slot[photo]
    img = store.images[s]
    H, W = img.shape[:2]
    y0, x0 = gy * TILE - margin, gx * TILE - margin
    size = TILE + 2 * margin
    ys = np.clip(np.arange(y0, y0 + size), 0, H - 1)
    xs = np.clip(np.arange(x0, x0 + size), 0, W - 1)
    # 写真外は端の画素を延長する
    out = np.asarray(img[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1])
    if out.shape[:2] != (size, size):
        out = out[np.ix_(ys - ys[0], xs - xs[0])]
    # 欠損タイル部分は対象タイルの平均色で埋める
    v = store.valid[s]
    if not v.all():
        out = out.copy()
        center = out[margin : margin + TILE, margin : margin + TILE]
        mean = center.reshape(-1, 3).mean(0).astype(np.uint8)
        for ty in range(max(gy - 1, 0), min(gy + 2, GRID_H)):
            for tx in range(max(gx - 1, 0), min(gx + 2, GRID_W)):
                if v[ty, tx]:
                    continue
                ry0 = max(ty * TILE - y0, 0)
                ry1 = min((ty + 1) * TILE - y0, size)
                rx0 = max(tx * TILE - x0, 0)
                rx1 = min((tx + 1) * TILE - x0, size)
                if ry0 < ry1 and rx0 < rx1:
                    out[ry0:ry1, rx0:rx1] = mean
    return out


if __name__ == "__main__":
    store = build_mosaics(force=True)
    print(store.images.shape, store.valid.sum(), len(store.photo_to_slot))
