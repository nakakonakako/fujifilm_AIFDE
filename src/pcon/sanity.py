"""データパイプラインの健全性チェック。例: uv run python -m src.pcon.sanity"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from .data import DATA_DIR, GRID_H, GRID_W, TILE, build_mosaics, crop_context, load_tables
from .train import OUT_DIR, TileDataset, dihedral, make_folds

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'NG'}] {name} {detail}")


def main():
    train, test = load_tables()
    store = build_mosaics()
    rng = np.random.default_rng(0)

    # 1. 件数・重複・ラベル
    check("train行数", len(train) == 10079, str(len(train)))
    check("test行数", len(test) == 8059, str(len(test)))
    check("trainとtestの写真が重ならない", not set(train.photo) & set(test.photo))
    check("ラベルは0/1のみ", set(train.label.unique()) <= {0, 1})
    check("trainのパス重複なし", train.path.is_unique)
    check("グリッド範囲", train.gx.between(0, GRID_W - 1).all() and train.gy.between(0, GRID_H - 1).all()
          and test.gx.between(0, GRID_W - 1).all() and test.gy.between(0, GRID_H - 1).all())
    check("ディレクトリ(CW/UW)とPコンラベルは別物", train.groupby(train.path.str[2:4]).label.mean().max() < 0.05)

    # 2. 復元：キャッシュ中のタイルが元ファイルと画素一致（全件のうち無作為300件 + _2 ファイル全件）
    allp = pd.concat([train, test], ignore_index=True)
    idx = list(rng.choice(len(allp), 300, replace=False)) + list(np.flatnonzero(allp.path.str.contains("_")))
    bad = 0
    for i in idx:
        r = allp.iloc[i]
        tile = np.asarray(Image.open(DATA_DIR / r.path).convert("RGB"))
        s = store.photo_to_slot[r.photo]
        got = np.asarray(store.images[s, r.gy * TILE:(r.gy + 1) * TILE, r.gx * TILE:(r.gx + 1) * TILE])
        bad += int(not np.array_equal(tile, got))
    check("キャッシュのタイルが元画像と一致", bad == 0, f"不一致 {bad}/{len(idx)}")

    # 3. 復元の配置：隣接タイル境界の画素差がランダムな位置の差と同程度（配置ミスなら大きくなる）
    ratios = []
    for ph, s in store.photo_to_slot.items():
        a = np.asarray(store.images[s, ::2]).astype(np.float32).mean(2)
        v = store.valid[s]
        dx = np.abs(np.diff(a, axis=1))
        seam = np.mean([dx[:, c * TILE - 1].mean() for c in range(1, GRID_W) if v[:, c - 1].all() and v[:, c].all()] or [0])
        inner = np.mean([dx[:, c * TILE + 100].mean() for c in range(GRID_W - 1)])
        ratios.append((seam / max(inner, 1e-6), ph))
    worst = max(ratios)
    check("全写真で横方向の境界が連続", worst[0] < 3.0, f"最大比 {worst[0]:.2f} (写真{worst[1]})")

    # 4. crop_context の中心が対象タイルそのもの
    bad = 0
    for i in rng.choice(len(train), 200, replace=False):
        r = train.iloc[i]
        for pad in ("edge", "mean"):
            c = crop_context(store, r.photo, r.gx, r.gy, 64, pad)
            s = store.photo_to_slot[r.photo]
            t = np.asarray(store.images[s, r.gy * TILE:(r.gy + 1) * TILE, r.gx * TILE:(r.gx + 1) * TILE])
            bad += int(c.shape != (384, 384, 3) or not np.array_equal(c[64:320, 64:320], t))
    check("切り出しの中心=対象タイル", bad == 0, f"不一致 {bad}")

    # 5. 欠損タイル：キャッシュ上で欠損している位置は存在しないファイル
    missing = {(p, gy, gx) for p, s in store.photo_to_slot.items() for gy, gx in zip(*np.nonzero(~store.valid[s]))}
    present = set(zip(allp.photo, allp.gy, allp.gx))
    check("欠損位置にファイルなし", not (missing & present), f"欠損 {len(missing)} 箇所")

    # 6. Dataset：検証時は決定的で中心がタイル、ラベルの対応が正しい
    ds = TileDataset(train.iloc[:50], store, 64, 384, train=False)
    x0, y0 = ds[7]
    x1, _ = ds[7]
    check("検証Datasetが決定的", bool((x0 == x1).all()))
    check("Datasetのラベル対応", all(ds[i][1] == train.label.iloc[i] for i in range(50)))
    ds_tr = TileDataset(train.iloc[:50], store, 64, 384, train=True, jitter=0)
    check("学習Dataset(jitter=0)の形状", tuple(ds_tr[3][0].shape) == (3, 384, 384))

    # 7. TTA の8変換がすべて異なり、可逆（k=0 は恒等）
    t = torch_rand()
    outs = [dihedral(t, k) for k in range(8)]
    check("dihedral k=0 は恒等", bool((outs[0] == t).all()))
    check("dihedral 8通りが相異なる", len({o.numpy().tobytes() for o in outs}) == 8)

    # 8. fold：写真単位で分割され、全実験で同一
    f = make_folds(train, 5, 42)
    check("foldが写真をまたがない", (pd.Series(f).groupby(train.photo.values).nunique() == 1).all())
    for e in ["base_r18_m64", "r18_m64_hard"]:
        p = OUT_DIR / e / "oof.csv"
        if p.exists():
            o = pd.read_csv(p)
            check(f"{e}: OOFの並び・fold・ラベルが一致", (o.path == train.path).all() and (o.fold.values == f).all()
                  and (o.label == train.label).all())
            tl = pd.read_csv(OUT_DIR / e / "test_logit.csv")
            check(f"{e}: テスト予測の並びが test.txt と一致", (tl.path == test.path).all())
            d = tl[tl.path.str.contains("7074-105")]
            check(f"{e}: 同一画像 7074-105/105_2 の予測が一致", d.logit.nunique() == 1, d.logit.round(4).tolist().__str__())

    print("\nALL OK" if ok else "\nNG あり")


def torch_rand():
    import torch
    return torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)


if __name__ == "__main__":
    main()
