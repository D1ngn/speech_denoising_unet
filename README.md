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
├── train.py
├── mk_data.py
└── data
		├── audio_files
				├── voice1000_noise_100

```







## 詳細

#### 使用データ

環境音：「[https://www.kaggle.com/mmoreaux/environmental-sound-classification-50/data#](https://www.kaggle.com/mmoreaux/environmental-sound-classification-50/data#)」

人の発話音声：「[https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)」





#### データの水増し法(Data Augumentation)

・





参考URL : 「[https://qiita.com/cvusk/items/61cdbce80785eaf28349](https://qiita.com/cvusk/items/61cdbce80785eaf28349)」

参考URL : 「[https://www.kaggle.com/CVxTz/audio-data-augmentation](https://www.kaggle.com/CVxTz/audio-data-augmentation)」

