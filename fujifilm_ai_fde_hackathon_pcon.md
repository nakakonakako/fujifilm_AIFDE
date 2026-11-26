# Pコン跡の認識

富士フイルム AI FDEハッカソン

**今回のコンテストでは、プログラムを「アップロード」した際に採点が行われます。**

**右上の「解答を提出する」ボタンはクリックしないでください。問題が提出済みとなり、再提出が出来なくなります。**

## 背景

富士フイルムでは様々な分野に対して画像技術による価値を提供しています。

- アルバム作成支援
  - https://www.fujifilm.com/jp/ja/consumer/photoprinting/album/year-album
- 医用画像診断
  - https://www.fujifilm.com/jp/ja/healthcare/healthcare-it/it-imaging
- 非破壊検査
  - https://www.fujifilm.com/jp/ja/business/inspection/non-destructive

富士フイルムの事業と研究開発の一端に触れていただくために、コンテストの題材として社会インフラ画像診断サービス「ひびみっけ」を取り上げます。
ひびみっけでは撮影した画像からひびわれなどを自動で抽出します。

https://www.fujifilm.com/jp/ja/business/inspection/infraservice/hibimikke

今回のコンテストでは、SDNET2018というデータセットを利用してひびみっけの研究開発を疑似的に体験していただきます。

## SDNET2018

SDNET2018は以下のサイトで配布されています。

https://digitalcommons.usu.edu/all_datasets/48/

"Download"（ページタイトル"SDNET2018: A concrete crack image dataset for machine learning applications"の右隣にあります）をクリックしてデータ（約500MB）をダウンロードしてください。

本問題で使う画像は、壁面（`W/`）の画像のみです。

## 問題概要

コンクリート施工時に生じるPコン跡の画像認識プログラムを作成してください。
具体的には、256x256の画像が入力されたときに、その画像にPコン跡が含まれていれば1、含まれていなければ0を出力するプログラムを作成してください。
Pコン跡の一部だけが画像の端に写っている場合や、モルタル等で埋め戻されたPコン跡が写っている場合も、「含まれている」とみなします。

## 配布物

- `train_label.csv`：学習用画像とその画像中のPコン跡の有無（1: あり、0: なし）の対応関係です。
- `test.txt`：評価用画像のリストです。
- `sample_submission.csv`：提出ファイルのサンプルです。

画像ファイルパスは、SDNET2018を展開したフォルダからの相対パス（例：`W/CW/7069-3.jpg`）です。
学習用画像と評価用画像は、撮影元の写真単位で分けています。同じ写真から切り出した画像が両方に含まれることはありません。

## 提出物

`test.txt` の各行に記載された画像ファイルパスに対して、Pコン跡の有無で1または0をカンマ区切りで出力したCSVファイルを提出してください。ヘッダー行は付けません。
`test.txt` のすべての画像について予測を記載してください。出力順は `test.txt` の画像ファイルパスの列挙順を推奨します。実際の出力例は `sample_submission.csv` を参照してください。

注意：最大アップロードサイズは10MBです。

## 補足

Pコン跡は沈みひびわれの原因になりうるため、外観検査においてPコン跡の周辺を確認しています。

- 参考文献1：鳥取県、コンクリート構造物ひび割れ抑制対策マニュアル（案）、2016
  - https://www.pref.tottori.lg.jp/secure/1021057/hibiwarema.pdf
- 参考文献2：中国地方整備局、コンクリート構造物の品質確保・向上の手引き（案）【施工編（監督・検査）】、2022
  - https://www.cgr.mlit.go.jp/ctc/pdf/technology/concrete/a4-07.pdf

また、画像からひびわれ抽出を行う際に、Pコン跡はひびわれと誤認識されて抽出される場合があり、例えば、誤抽出低減のためにPコン跡検出器を作成・活用した研究があります。

- 参考文献3：全ら、画像解析によるひび割れ検出とハフ変換による型枠跡およびPコン跡の同定、2015
  - http://data.jci-net.or.jp/data_pdf/37/037-01-1291.pdf

## 評価方法

F1スコアで評価します。

評価用画像は、開催期間中にスコアを表示するための画像（public）と、期間終了後の最終評価に使う画像（private）に分かれています。
アップロード時に表示されるスコアは public の画像のみで計算したものです。最終評価は private の画像で行うため、表示されるスコアと最終評価のスコアは一致しません。
どの画像が public / private に属するかは公開しません。

```
f1_score = 2 * (recall * precision) / (recall + precision)

recall = true_positive / (true_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
```

例えば、Pythonでスコアを計算するコードは、提出するCSVファイルを `sample_submission.csv` とし、正解が記載されたCSVファイルを `test_label.csv` とすると、以下の通りです。

```python
import pandas as pd
from sklearn.metrics import f1_score


def read_csv(path):
    df = pd.read_csv(path, header=None)
    return df[0].to_list(), df[1].values


if __name__ == "__main__":
    X0, y_true = read_csv("test_label.csv")
    X1, y_pred = read_csv("sample_submission.csv")
    pred = dict(zip(X1, y_pred))

    print(f1_score(y_true, [pred[x] for x in X0]))
```

## 禁止事項

- 配布物・SDNET2018以外の外部データの使用。
  - ImageNetで事前学習したモデルやGoogle Cloud APIなどのAPIの使用もNG。
- 評価用画像に対して手動でラベル付けした結果の提出。
- 開催期間中の、外部への質問およびコードの共有など、いわゆるPrivate Sharingに該当する行為。

## クレジット表示

本問題のデータはCreative Commons Attribution 4.0 License (CC BY 4.0) に基づいて利用しています。

Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image dataset for machine learning applications. Utah State University.
https://doi.org/10.15142/T3TD19
