"""学習済みモデルの中身の分析: 層ごとの重み統計と Grad-CAM。

例: uv run python -m src.pcon.inspect_model --exp r18_m64_hard_s7 --tiles 7085-232 7122-4 7128-162 --out /tmp/cam.jpg
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import timm
import torch

from .data import build_mosaics, crop_context, load_tables
from .train import OUT_DIR, make_folds, sigmoid, to_input


def weight_stats(model: torch.nn.Module) -> pd.DataFrame:
    rows = []
    for name, p in model.named_parameters():
        if p.ndim < 1:
            continue
        w = p.detach().float().flatten()
        rows.append(dict(name=name, n=w.numel(), mean=w.mean().item(), std=w.std().item() if w.numel() > 1 else 0,
                         absmax=w.abs().max().item(), near0=(w.abs() < 1e-3).float().mean().item()))
    return pd.DataFrame(rows)


def gradcam(model, x, layer):
    feats, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    logit = model(x).squeeze()
    logit.backward()
    h1.remove()
    h2.remove()
    a, g = feats["a"][0], grads["g"][0]
    cam = torch.relu((g.mean((1, 2), keepdim=True) * a).sum(0)).detach().cpu().numpy()
    return logit.item(), cam / (cam.max() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--tiles", nargs="+", required=True, help="写真ID-番号")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exp_dir = OUT_DIR / a.exp
    args = SimpleNamespace(**json.loads((exp_dir / "args.json").read_text()))
    train, test = load_tables()
    train["fold"] = make_folds(train, args.n_folds, seed=42)
    store = build_mosaics()

    models = []
    for k in range(args.n_folds):
        m = timm.create_model(args.arch, pretrained=False, num_classes=1)
        m.load_state_dict(torch.load(exp_dir / f"model_f{k}.pt", map_location="cpu"))
        models.append(m.eval())

    # 重み統計（fold 間で比較）
    ws = [weight_stats(m).set_index("name") for m in models]
    summ = pd.DataFrame({"std_f0": ws[0]["std"], "std_f1": ws[1]["std"], "near0_f0": ws[0]["near0"], "n": ws[0]["n"]})
    conv = summ[[("conv" in n or "fc" in n) and n.endswith("weight") for n in summ.index]]
    print("畳み込み層・全結合層の重み（fold0/fold1 の標準偏差、ほぼ0の割合）")
    print(conv.round(4).to_string())
    bn = ws[0][[n.endswith("bn2.weight") or n.endswith("bn1.weight") for n in ws[0].index]]
    print("\nBN の γ: 平均 %.3f, γ<0.05 の割合（使われていないチャネル）%.3f" % (
        np.mean([models[0].state_dict()[n].mean().item() for n in bn.index]),
        np.mean([(models[0].state_dict()[n].abs() < 0.05).float().mean().item() for n in bn.index])))

    allp = pd.concat([train.assign(split="train"), test.assign(split="test", fold=-1)], ignore_index=True)
    cells = []
    for t in a.tiles:
        ph, idx = t.split("-")
        r = allp[(allp.photo == ph) & (allp["index"] == int(idx))].iloc[0]
        img = crop_context(store, r.photo, int(r.gx), int(r.gy), args.margin)
        x = to_input(torch.from_numpy(img).permute(2, 0, 1)[None], "cpu")
        use = [models[int(r.fold)]] if r.split == "train" else models  # 学習画像はそれを見ていない fold のモデル
        cams, logits = [], []
        for m in use:
            lg, cam = gradcam(m, x.clone().requires_grad_(False), m.layer4)
            cams.append(cam)
            logits.append(lg)
        cam = cv2.resize(np.mean(cams, 0), (img.shape[1], img.shape[0]))
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        vis = (0.55 * img + 0.45 * heat).astype(np.uint8)
        m0 = args.margin
        cv2.rectangle(vis, (m0, m0), (m0 + 255, m0 + 255), (255, 255, 255), 2)
        lab = f"{t} {r.split} y={r.label} p={sigmoid(np.mean(logits)):.2f}"
        both = np.vstack([np.hstack([img, vis]), np.full((24, img.shape[1] * 2, 3), 255, np.uint8)])
        cv2.putText(both, lab, (4, both.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cells.append(both)
        print(lab)
    cv2.imwrite(a.out, cv2.cvtColor(np.vstack(cells), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
