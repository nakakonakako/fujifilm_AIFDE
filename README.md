# Pコン跡の認識（富士フイルム AI FDEハッカソン）

問題文は `fujifilm_ai_fde_hackathon_pcon.md` を参照。

## 動作環境

- Windows 11 + WSL2 (Ubuntu), NVIDIA GeForce RTX 3060 Ti (8GB), CPU 20コア, メモリ 16GB
- Python 3.12, [uv](https://docs.astral.sh/uv/)
- PyTorch 2.14.1 (CUDA 13.0), timm 1.0.31（依存関係は `pyproject.toml` / `uv.lock`）
  - CUDA 12.x ドライバの環境では `pyproject.toml` の `cu130` を `cu126` に変えて `uv lock && uv sync`
- 事前学習済みモデル・外部データは不使用（すべてランダム初期化から学習）

## 再現手順（最終提出 v21: `submissions/21_ens_stripe+best_pos1.92_interim.csv`）

```bash
uv sync
# SDNET2018 を展開し data/DATA_Maguire_20180517_ALL/W/... となるように配置
bash scripts/run_v21.sh   # 学習（15実験）→ submission.csv 作成
```

詳細は `docs/SOLUTION.md` を参照。GPU の非決定的な演算（cuDNN）を使っているため、再実行時の予測は数タイル程度ずれることがあります。

## 構成

- `src/pcon/data.py`: ファイル名（`写真ID-タイル番号[_2].jpg`）から 18x14 タイルの元写真を復元し memmap にキャッシュ。対象タイル＋周辺の切り出し
- `src/pcon/train.py`: 写真単位 StratifiedGroupKFold での学習・OOF/テスト予測（オプション: 難例重み付け、疑似ラベル、全データ学習ほか）
- `src/pcon/make_sub.py` / `ensemble.py` / `scripts/final_sub.py`: 提出CSV作成・アンサンブル
- `src/pcon/compare.py`: 写真単位ブートストラップによる実験比較
- `src/pcon/errors.py` / `inspect_model.py`: エラー分析・Grad-CAM
- `src/pcon/sanity.py`: データパイプラインの健全性チェック
- `scripts/queue_worker.sh` / `sync_lab.sh`: GPUジョブキュー・リモート結果回収
- `submissions/`: 提出CSVと記録（`LOG.md`）
