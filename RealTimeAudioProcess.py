# -*- coding:utf-8 -*-

# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os
import numpy as np
import pyaudio
import wave
import struct
import librosa

from train import Unet
from mk_data import wave_to_spec, save_audio_file
from inference import spec_to_wav


def write_wave(idx, data, sr, save_dir): # sr:サンプリング周波数
    sin_wave = [int(x * 32767.0) for x in data]
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    wav_file_path = os.path.join(save_dir, str(idx)+'.wav')
    w = wave.Wave_write(wav_file_path)
    p = (1, 2, sr, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()


# 人の声のスペクトログラムを抽出
def extract_voice_spec(net, mixed_mag, mixed_phase):
    # スペクトログラムを正規化
    max_mag = mixed_mag.max()
    normed_mixed_mag = mixed_mag / max_mag
    # データの形式をモデルに入力できる形式に変更する
    # librosaを使って、スペクトログラムを算出すると周波数要素が513になるので、512にスライス
    mag_sliced = normed_mixed_mag[1:, :]
    phase_sliced = mixed_phase[1:, :]
    # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
    mag_padded = np.pad(mag_sliced, [(0, 0), (0, 128 - mixed_mag.shape[1])], 'constant')
    phase_padded = np.pad(phase_sliced, [(0, 0), (0, 128 - mixed_mag.shape[1])], 'constant')
    # 0次元目と1次元目に次元を追加
    mag_expanded = mag_padded[np.newaxis, np.newaxis, :, :] # shape:(1, 1, 512, 128)
    phase_expanded = phase_padded[np.newaxis, np.newaxis, :, :]
    # numpy形式のデータをpytorchのテンソルに変換
    mag_tensor = torch.from_numpy(mag_expanded)
    # 環境音のmaskを計算
    mask = net(mag_tensor)
    # pytorchのtensorをnumpy配列に変換
    mask = mask.detach().numpy()
    # 人の声を取り出す
    normed_separated_voice_mag = mask * mag_expanded
    # 正規化によって小さくなった音量を元に戻す
    separated_voice_mag = normed_separated_voice_mag * max_mag
    # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
    voice_spec = separated_voice_mag * phase_expanded  # shape:(1, 1, 512, 128)
    voice_spec = np.squeeze(voice_spec) # shape:(512, 128)
    voice_spec = voice_spec[:, :mixed_mag.shape[1]] # 入力と同じ大きさのスペクトログラムに戻す

    return voice_spec



if __name__ == "__main__":

    # 使用するパラメータの設定
    N = 10
    CHUNK = 1024 * N # １度に処理する音声のサンプル数
    RATE = 16000 # サンプリングレート
    CHANNELS = 1
    fft_size = 1024 # 高速フーリエ変換のフレームサイズ
    hop_length = 768 # 高速フーリエ変換におけるフレーム間のオーバーラップ長
    audio_idx = 0
    wave_dir = "./audio_data/rec/"
    os.makedirs(wave_dir, exist_ok=True)

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    checkpoint_path = "./ckpt/ckpt_epoch50.pt"
    # ネットワークモデルを指定
    net = Unet()
    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 学習済みのパラメータをロード
    net_params = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(net_params['state_dict'])
    # Unetを使って推論
    # ネットワークを推論モードへ
    net.eval()

    p = pyaudio.PyAudio()

    stream = p.open(format = pyaudio.paInt16,
    		channels = CHANNELS,
    		rate = RATE,
    		frames_per_buffer = CHUNK,
    		input = True,
    		output = True) # inputとoutputを同時にTrueにする

    # マイクで取得した音声に対して音源分離処理を行う
    while stream.is_active():
        print(audio_idx)
        input = stream.read(CHUNK)
        # input = voice_separate(input) # 音源分離処理を行う
        data = np.frombuffer(input, dtype="int16") / 32768.0 # 量子化ビット数が16bitの場合の正規化
        # マルチチャンネルで行う場合
        # チェンネル1のデータはmultichannel_data[:, 0]、チャンネル2のデータはmultichannel_data[:, 1]...
        # chunk_length = len(data) / CHANNELS
        # multichannel_data = np.reshape(data, (chunk_length, CHANNELS))
        # 音声データをスペクトログラムに変換
        mixed_mag, mixed_phase = wave_to_spec(data, fft_size, hop_length) # wavをスペクトログラムへ
        # マイクで取得した音声のスペクトログラムから人の声のスペクトログラムを抽出
        voice_spec = extract_voice_spec(net, mixed_mag, mixed_phase)
        # スペクトログラムを音声データに変換
        masked_voice_data = spec_to_wav(voice_spec, hop_length)
        # オーディオファイルとそのスペクトログラムを保存
        masked_voice_path = "./output/test/masked_voice{}.wav".format(audio_idx)
        save_audio_file(masked_voice_path, masked_voice_data, sampling_rate=RATE)

        # data =[]
        # data = np.frombuffer(input, dtype="int16") / 32768.0 # 量子化ビット数が16bitの場合の正規化
        # write_wave(audio_idx, data, RATE, wave_dir) # 処理後の音声データをwavファイルとして保存

        # output = stream.write(input)

        audio_idx += 1

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Stop")
