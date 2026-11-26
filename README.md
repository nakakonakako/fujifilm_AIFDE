# Pコン跡の認識（富士フイルム AI FDEハッカソン）

問題文は `fujifilm_ai_fde_hackathon_pcon.md` を参照。

## 動作環境（提出 v11 を作成した環境）

- Windows 11 + WSL2 (Ubuntu), NVIDIA GeForce RTX 3060 Ti (8GB), CPU 20コア, メモリ 16GB
- Python 3.12, [uv](https://docs.astral.sh/uv/)
- PyTorch 2.14.1 (CUDA 13.0), timm 1.0.31（依存関係は `pyproject.toml` / `uv.lock`）
  - CUDA 12.x ドライバの環境では `pyproject.toml` の `cu130` を `cu126` に変えて `uv lock && uv sync`
- 事前学習済みモデル・外部データは不使用（すべてランダム初期化から学習）

## 再現手順（提出 v11: `submissions/11_r18_m64_hard_s7_oof0.880.csv`）

```bash
# 1. 環境構築
uv sync

# 2. データ配置: SDNET2018 を展開し data/DATA_Maguire_20180517_ALL/W/... となるようにする
#    （DATA_Maguire_20180517_ALL.zip を data/ で展開し、中の SDNET2018.zip も同じ場所で展開）

# 3. タイルから元写真（18x14タイル）を復元したキャッシュを作成（data/cache/, 約3.5GB）
uv run python -m src.pcon.data

# 4. ベースラインの5-fold学習（OOF予測を難例重み付けに使う）  RTX 3060 Ti で約60分
uv run python -m src.pcon.train --exp base_r18_m64

# 5. 難例重み付けモデルの5-fold学習（seed 7）  約60分
uv run python -m src.pcon.train --exp r18_m64_hard_s7 --hard-from base_r18_m64 --seed 7 --epochs 30 --eval-every 5

# 6. 提出CSV作成（5 foldモデルの平均、閾値はOOFのF1最大点 = 0.706）
uv run python -m src.pcon.make_sub --exps r18_m64_hard_s7 --out submission.csv
uv run python -m src.pcon.check_submission submission.csv
```

GPU の非決定的な演算（cuDNN）を使っているため、再実行時の予測は数タイル程度ずれることがあります。

## 構成

- `src/pcon/data.py`: ファイル名（`写真ID-タイル番号[_2].jpg`）から 18x14 タイルの元写真を復元し memmap にキャッシュ。対象タイル＋周辺の切り出し
- `src/pcon/train.py`: 写真単位 StratifiedGroupKFold での学習・OOF/テスト予測（オプション: 難例重み付け、疑似ラベル、全データ学習ほか）
- `src/pcon/make_sub.py` / `ensemble.py` / `scripts/final_sub.py`: 提出CSV作成・アンサンブル
- `src/pcon/compare.py`: 写真単位ブートストラップによる実験比較
- `src/pcon/errors.py` / `inspect_model.py`: エラー分析・Grad-CAM
- `src/pcon/sanity.py`: データパイプラインの健全性チェック
- `scripts/queue_worker.sh` / `sync_lab.sh`: GPUジョブキュー・リモート結果回収
- `submissions/`: 提出CSVと記録（`LOG.md`）
