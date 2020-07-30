# 雑音環境下における話者の声の分離



## 概要

人の声と環境音が混ざった音声から人の声を取り出すモデルを作成する。元となるモデルには[unet](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf)を使用する。



#### 注意点

・PyTorchはパラメータを持つ層とそうでない層を特に強く区別しない

→基本的には全てクラスとして、\__init__で準備しておき、forwardでメソッドとして演算を呼び出すというのが定石

・学習を容易にするためにスペクトログラムを[0,1]の範囲に正規化する

→その際オリジナル音源とボーカルパートの数値比を変えないよう注意する必要がある(ボーカルの音量を一致させる為)。同じ数値で割る。
・スペクトログラムからランダムに一定長分切り取った部分をNNに入力して学習を行います。



## ディレクトリ構成

```
voice_separation
├── README.md
├── mk_data.py (データ作成用プログラム)
├── train.py (学習用プログラム)
├── inference.py (学習済みモデル実行用プログラム)
├── RealTimeAudioProcess.py (マイクロホンで取得した音声に対してリアルタイムで雑音除去を行う)
└── data
    ├── jvs_ver1 (複数の日本人の発話データ)
    ├── environmental-sound-classification-50 (環境音の音声データ)
    ├── voice1000_noise_100 (混合音声の音声データとスペクトログラムデータ) ← mk_data.pyにより生成

```







## 実行手順

#### 使用データ

下記のURLから人の発話データと環境音の音声データをダウンロードし、`./data`内に格納

人の発話データ：「[https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)」

環境音の音声データ：「[https://www.kaggle.com/mmoreaux/environmental-sound-classification-50/data#](https://www.kaggle.com/mmoreaux/environmental-sound-classification-50/data#)」



#### データ作成

```
$ python3 mk_data.py
```

を実行すると、`voice[人の発話データの数]_noise_[環境音の音声データの数]`といったディレクトリに学習用のデータと検証用のデータ、評価用のデータが作成される。



#### 学習の開始

```
$ python3 train.py
```

を実行すると学習が開始される。





#### 学習の再開

途中まで学習を行ったモデルのパラメータ(チェックポイントファイル)を使用して学習を再開する場合は、以下のように既存のチェックポイントファイルのパスを指定する。

```
$ python3 train.py --checkpoint_path="./ckpt/ckpt_epoch50.pt"
```



#### マイクロホンで取得した音声に対してリアルタイム雑音除去

マイクロホンで取得した音声に対してリアルタイムで雑音除去を行う。

```
$ python3 RealTimeAudioProcess.py 
```

時間を指定して録音する場合は以下のように引数に録音時間[s]を指定する。

```
$ python3 RealTimeAudioProcess.py -r 5
```















## 詳細

#### データの水増し法(Data Augumentation)

人の発話音声それぞれに対して、環境音を足し合わせデータを作成する

(例)人の発話音声100種類 × 環境音10種類 = データセット1000種類



参考URL : 「[https://qiita.com/cvusk/items/61cdbce80785eaf28349](https://qiita.com/cvusk/items/61cdbce80785eaf28349)」

参考URL : 「[https://www.kaggle.com/CVxTz/audio-data-augmentation](https://www.kaggle.com/CVxTz/audio-data-augmentation)」





## やるべきこと

#### ネットワークモデルに関して

✔︎学習を途中から再開できるようにする

・モデルをnn.Sequentialで見やすく書き直す

✖️入力をスペクトログラムから**波形データ**や**対数メルスペクトログラム**に変換して精度が向上するか調査を行う

✖️CNNの部分を**dilated CNN(拡張畳み込み)**に変更して精度が向上するか調査を行う

・学習にかかったトータルの時間をログに書き出すようにする

#### 音源分離性能の評価に関して

✔︎元の音声ファイルをコピーしてoutputディレクトリに保存

・音源分離の精度を定量的に評価できるようにする

・学習データ量を増やして、精度が向上するか調査を行う(10000文でうまくいっているという論文あり)

・自分の声を大量に学習させて(JSUTを1000文くらい読んでみる)、自分の声のみを分離するモデルも作ってみる

・環境音の大きさを変化させて分離性能を評価

・バンドパスフィルタを使用した雑音除去モデルとの比較も行う

→エアコン、PC、冷蔵庫、テレビ、スマホなどの家電製品から発生する音は比較的低周波(0~5kHzに分布)であるのに対して、人の呼吸音は幅広い範囲に分布

#### ロボットへの実装に関して

✔︎マイクロホンで取得した音声に対してリアルタイムで雑音除去できるようにする

・音源分離を8chマイクロホンに対応させる

→「8つの入力を並列処理で音源分離 」or 「1つの入力だけ音源分離してその情報だけを音声認識」

→前者は音源定位性能と音声認識性能の両方が上がる可能性がある分、スペック的な問題で難しい？

→後者はスペック的には問題ないが、音源定位性能自体は向上せず、音声認識性能のみの向上になる

・シングルチャンネルの音声からマルチチャンネル(8ch)の音声データセットを作れないか調査してみる

・発話区間を検出するモデルを組み込む

→unetへの入力が固定長の音声のスペクトログラムであるため



## 参考文献

・DNNを用いたマルチチャンネル音源分離「Multichannel Audio Source Separation With Deep Neural Networks」





