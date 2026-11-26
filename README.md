# Pコン跡の認識（富士フイルム AI FDEハッカソン）

問題文は `fujifilm_ai_fde_hackathon_pcon.md` を参照。

## セットアップ

```bash
uv sync
# SDNET2018 を data/ に展開（data/DATA_Maguire_20180517_ALL/W/... となる配置）
uv run python -m src.pcon.data   # 写真の復元キャッシュを data/cache/ に作成
```

## 構成

- `src/pcon/data.py`: ファイル名（`写真ID-タイル番号[_2].jpg`）から 18x14 タイルの元写真を復元し、memmap にキャッシュ。周囲を含めた切り出しを提供
- `src/pcon/train.py`: 写真単位 StratifiedGroupKFold で学習（事前学習なし）。OOF・テスト予測・提出CSVを `outputs/<exp>/` に保存
- `src/pcon/post.py`: 近傍タイルの予測を特徴量にした後処理（ロジスティック回帰）

## 実行例

```bash
uv run python -m src.pcon.train --exp base_r18_m64 --arch resnet18d --margin 64
uv run python -m src.pcon.post --exps base_r18_m64 --out post_base
```
