# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import librosa

from train import Unet
from mk_data import load_audio_file, save_audio_file



# 音声データをスペクトログラムに変換する
def wav_to_spec_for_inference(data, fft_size, hop_length):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    spec = librosa.stft(data, n_fft=fft_size, hop_length=hop_length)
    mag = np.abs(spec) # 振幅スペクトログラムを取得
    phase = np.exp(1.j * np.angle(spec)) # 位相スペクトログラムを取得(フェーザ表示)
    return mag, phase




if __name__ == '__main__':

    # 学習済みの重みのパスを指定
    weights_path = "./ckpt/ckpt_epoch{}.pt"
    # ネットワークモデルを指定
    net = Unet()
    # GPUが使える場合あはGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 学習済みの重みをロード
    net_weights = torch.load(weights_path, map_location=device)
    net.load_state_dict(net_weights)
    # Unetを使って推論
    # ネットワークを推論モードへ
    net.eval()

    # 音声ファイルのパスを指定
    input_audio_file = "./data/voice1000_noise100/test/BASIC5000_1017_001_mixed.wav"
    # 音声データをロード(現在は学習時と同じ処理をしているが、いずれはマイクロホンのリアルストリーミング音声を入力にしたい)
    sampling_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 5 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適化かも？
    input_audio_data = load_audio_file(input_audio_file, audio_length, sampling_rate)
    #　音声データをスペクトログラムに変換
    fft_size = 1024 # 高速フーリエ変換のフレームサイズ
    hop_length = 768 # 高速フーリエ変換におけるフレーム間のオーバーラップ長
    input_mag, input_phase = wav_to_spec_for_inference(input_audio_data, fft_size, hop_length) # wavをスペクトログラムへ
    # スペクトログラムを正規化
    max_mag = input_mag.max()
    normed_input_mag = input_mag / max_mag
    # データの形式をモデルに入力できる形式に変更する
    # librosaを使って、スペクトログラムを算出すると周波数要素が513になるので、512にスライス
    mag_sliced = normed_input_mag[1:, :]
    # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
    mag_padded = np.pad(mag_sliced, [(0, 0), (0, 128 - mag_sliced.shape[1])], 'constant')
    # 0次元目と1次元目に次元を追加
    mag_expanded = mag_padded[np.newaxis, np.newaxis, :, :] # shape:(1, 1, 512, 128)

    # 環境音のmaskを計算
    mask = net(mag_expanded)
    # 人の声を取り出す
    separated_voice_mag = mask * mag_expanded
    # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
    voice_spec = separated_voice_mag * input_phase
    output_audio_data = librosa.istft(voice_spec, hop_length=hop_length)
    # オーディオファイルを保存
    save_path = "./output/masked_voice.wav"
    save_audio_file(save_path, output_audio_data, sampling_rate=16000)
