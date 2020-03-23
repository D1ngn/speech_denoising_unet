# 雑音環境下における話者の声の分離



## 概要

人の声と環境音が混ざった音声から人の声を取り出すモデルを作成する。モデルにはunetを使用する。



#### 注意点

・PyTorchはパラメータを持つ層とそうでない層を特に強く区別しません。基本的には全てクラスとして、__init__で準備しておき、fowardでメソッドとして演算を呼び出すというのが定石

・学習を容易にするためにスペクトログラムを[0,1]の範囲に正規化するのですが、その際オリジナル音源とボーカルパートの数値比を変えないよう注意する必要があります(ボーカルの音量を一致させる為)。同じ数値で割ること。
そのスペクトログラムからランダムに一定長分切り取った部分をNNに入力して学習を行います。



## ディレクトリ構成

```
voice_separation
├── README.md
├── mk_data.py (データ作成用プログラム)
├── train.py (学習用プログラム)
├── inference.py (学習済みモデル実行用プログラム)
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

途中まで学習を行ったモデルのパラメータ(チェックポイントファイル)を使用して学習を再開する場合は、













## 詳細

#### データの水増し法(Data Augumentation)

人の発話音声それぞれに対して、環境音を足し合わせデータを作成する

(例)人の発話音声100種類 × 環境音10種類 = データセット1000種類



参考URL : 「[https://qiita.com/cvusk/items/61cdbce80785eaf28349](https://qiita.com/cvusk/items/61cdbce80785eaf28349)」

参考URL : 「[https://www.kaggle.com/CVxTz/audio-data-augmentation](https://www.kaggle.com/CVxTz/audio-data-augmentation)」



