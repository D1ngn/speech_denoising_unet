# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wave
import subprocess
import time

from natsort import natsorted
from tqdm import tqdm
from train import Unet
from mk_data import load_audio_file, save_audio_file, wave_to_spec

# # 音声評価用
# import sys
# sys.path.append('..')
# from MyLibrary.MyFunc import wave_plot, audio_eval

# モデルのパラメータ数をカウント
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# スペクトログラムを音声データに変換する
def spec_to_wav(spec, hop_length):
    # 逆短時間フーリエ変換(iSTFT)を行い、スペクトログラムから音声データを取得
    wav_data = librosa.istft(spec, hop_length=hop_length)
    return wav_data


# スペクトログラムを図にプロットする関数 今は使っていない
def spec_plot_old(input_spec, save_path):
    # パワースペクトルを対数パワースペクトルに変換
    log_power_spec = librosa.amplitude_to_db(input_spec, ref=np.max)
    plt.figure(figsize=(12,5)) # 図の大きさを指定
    librosa.display.specshow(log_power_spec, x_axis='time', y_axis='log')
    plt.title('Spectroram') # タイトル
    plt.xlabel("time[s]") # 横軸のラベル
    plt.ylabel("amplitude") # 縦軸のラベル
    plt.colorbar(format='%+02.0f dB') # カラーバー表示
    plt.savefig(save_path)

# スペクトログラムを図にプロットする関数
def spec_plot(base_dir, wav_path, save_path, audio_length):
    # soxコマンドによりwavファイルからスペクトログラムの画像を生成
    cmd1 = "sox {} -n trim 0 {} rate 16.0k spectrogram".format(wav_path, audio_length)
    subprocess.call(cmd1, shell=True)
    # 生成されたスペクトログラム画像を移動
    #(inference.pyを実行したディレクトリにスペクトログラムが生成されてしまうため)
    spec_path = os.path.join(base_dir, "spectrogram.png")
    cmd2 = "mv {} {}".format(spec_path, save_path)
    subprocess.call(cmd2, shell=True)

if __name__ == '__main__':

    sampling_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 3 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適かも？
    fft_size = 1024 # 高速フーリエ変換のフレームサイズ
    hop_length = 768 # 高速フーリエ変換におけるフレーム間のオーバーラップ長
    spec_frame_num = 64 # スペクトログラムのフレーム数 spec_freq_dim=512のとき、音声の長さが5秒の場合は128, 3秒の場合は64

    # 音声データをロード(現在は学習時と同じ処理をしているが、いずれはマイクロホンのリアルストリーミング音声を入力にしたい)
    mixed_audio_path = "./istft.wav"
    mixed_audio_data = load_audio_file(mixed_audio_path, audio_length, sampling_rate)
    #　音声データをスペクトログラムに変換
    mixed_mag, mixed_phase = wave_to_spec(mixed_audio_data, fft_size, hop_length) # wavをスペクトログラムへ
    # スペクトログラムを正規化
    max_mag = mixed_mag.max()
    normed_mixed_mag = mixed_mag / max_mag
    # データの形式をモデルに入力できる形式に変更する
    # librosaを使って、スペクトログラムを算出すると周波数要素が513になるので、512にスライス
    # mag_sliced = normed_mixed_mag[:512, :]
    # phase_sliced = mixed_phase[:512, :]
    mag_sliced = normed_mixed_mag[1:, :]
    phase_sliced = mixed_phase[1:, :]
    # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
    mag_padded = np.pad(mag_sliced, [(0, 0), (0, spec_frame_num - mixed_mag.shape[1])], 'constant')
    phase_padded = np.pad(phase_sliced, [(0, 0), (0, spec_frame_num - mixed_mag.shape[1])], 'constant')
    # 0次元目と1次元目に次元を追加
    mag_expanded = mag_padded[np.newaxis, np.newaxis, :, :] # shape:(1, 1, 512, 128)
    phase_expanded = phase_padded[np.newaxis, np.newaxis, :, :]
    # 正規化によって小さくなった音量を元に戻す
    separated_voice_mag = mag_expanded * max_mag
    # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
    voice_spec = separated_voice_mag * phase_expanded  # shape:(1, 1, 512, 128)
    voice_spec = np.squeeze(voice_spec) # shape:(512, 128)
    estimated_voice_data = spec_to_wav(voice_spec, hop_length)
    # オーディオデータを保存
    estimated_voice_path = "./estimated_voice.wav"
    save_audio_file(estimated_voice_path, estimated_voice_data, sampling_rate=16000)
