# 解答の説明（提出 v11）

対象の提出: `submissions/11_r18_m64_hard_s7_oof0.880.csv`（public 0.9333）

## ソースコード

https://github.com/nakakonakako/fujifilm_AIFDE/tree/feat/pcon-baseline

## 解答手順

### 動作環境

- Windows 11 + WSL2（Ubuntu）、NVIDIA GeForce RTX 3060 Ti（8GB）、CPU 20コア、メモリ 16GB
- Python 3.12、uv（パッケージ管理）
- PyTorch 2.14.1（CUDA 13.0）、timm 1.0.31、scikit-learn、OpenCV、pandas（`pyproject.toml` / `uv.lock` で固定）
  - CUDA 12.x ドライバの環境では `pyproject.toml` の `cu130` を `cu126` に置き換えて `uv lock && uv sync`
- 事前学習済みモデル・外部データは不使用（すべてランダム初期化から学習）

### 手順

```bash
# 1. 環境構築
uv sync

# 2. データ配置
#    SDNET2018 の DATA_Maguire_20180517_ALL.zip を data/ で展開し、
#    中の SDNET2018.zip も同じ場所で展開（data/DATA_Maguire_20180517_ALL/W/... となる配置）

# 3. タイルから元写真（横18×縦14タイル）を復元したキャッシュを作成（data/cache/, 約3.5GB）
uv run python -m src.pcon.data

# 4. ベースラインの5-fold学習（約60分）。OOF予測を手順5の難例重み付けに使う
uv run python -m src.pcon.train --exp base_r18_m64

# 5. 難例重み付けモデルの5-fold学習（seed 7, 約60分）
uv run python -m src.pcon.train --exp r18_m64_hard_s7 --hard-from base_r18_m64 --seed 7 --epochs 30 --eval-every 5

# 6. 提出CSV作成（5 foldモデルの平均、閾値はOOFのF1最大点 0.706）
uv run python -m src.pcon.make_sub --exps r18_m64_hard_s7 --out submission.csv
uv run python -m src.pcon.check_submission submission.csv
```

- 所要時間は RTX 3060 Ti で合計約2時間
- GPU の非決定的演算（cuDNN）を使うため、再実行時に予測が数タイル程度ずれることがある

### 主な設定

| 項目 | 値 |
|---|---|
| モデル | ResNet18-D（timm `resnet18d`）、ランダム初期化、出力1ch（BCE） |
| 入力 | 対象タイル + 周囲64px（384×384） |
| 学習 | AdamW（lr 2e-3, weight decay 0.05）、cosine学習率（2エポックのwarmup）、30エポック、batch 48、bf16、EMA 0.999 |
| サンプリング | 陽性約15%になる重み付きサンプリング × 難例重み（最大6倍） |
| データ拡張 | 反転・90°回転の8通り、平行移動±24px、明るさ・コントラスト・色・グレー化、ぼかし、ノイズ |
| 推論 | 8通りのTTA、5 foldモデルの平均 |
| 検証 | 撮影元の写真単位の5-fold（StratifiedGroupKFold） |

## 創意工夫

1. **タイルから元写真を復元して周辺情報を入力に使用**
   ファイル名（`写真ID-タイル番号.jpg`）が元写真内の位置を表すことに着目し、256pxのタイルを横18×縦14に並べ直して元写真を完全に復元しました。Pコン跡はタイル境界をまたぐことが多く（陽性タイルの84%は隣接タイルも陽性）、ルール上「端に一部だけ写っていても陽性」のため、対象タイルに周囲64pxを加えて入力し、穴の本体が隣のタイルにあるケースでも判定できるようにしました。評価用画像も同様に復元しています。

2. **写真単位の交差検証による提出前評価**
   学習用と評価用が写真単位で分かれていることに合わせ、写真単位の5-foldで検証しました。写真単位ブートストラップで改善が偶然でないかを確かめながら実験を進め、検証スコア（OOFのAP）とpublicスコアの順位相関は0.86となり、提出前に良し悪しを判断できるようにしました。

3. **難例重み付け（ハードエグザンプルマイニング）**
   ベースラインのOOF予測で間違えやすかったタイル（境界ぎりぎりのPコン跡、靴などの暗い物体、気泡など）を最大6倍の頻度で学習させました。OOF予測はそのタイルを学習していないモデルの出力なので、難しさを公平に測れます。この施策でpublicスコアがベースラインの0.871から0.921〜0.933に改善しました。

4. **エラー分析とモデルの可視化に基づく改善**
   見逃し・誤検出を周辺込みの画像で一覧化して分類（境界ぎりぎり、遠くの小さなPコン跡、暗い物体、写真端のパディングによる模様など）し、Grad-CAMでモデルが「明暗の境界の曲線」を手がかりにしていることを確認したうえで、データ拡張や入力の設計を見直しました。

5. **事前学習なしの条件に合わせた設計**
   小規模データでは大きなモデル（ResNet34）よりResNet18が良いことを検証で確認し、EMA・データ拡張・TTAでランダム初期化からの学習を安定させました。

## 実験の記録

すべての提出とスコアは `submissions/LOG.md` を参照。
