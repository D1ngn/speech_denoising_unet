# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wave
import subprocess
import time

# from train import Unet
from models import Unet_kernel3
from mk_data import load_audio_file, save_audio_file, wave_to_spec

# 音声評価用
import sys
sys.path.append('..')
from MyLibrary.MyFunc import wave_plot, audio_eval

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
    fft_size = 512 # 高速フーリエ変換のフレームサイズ
    hop_length = 160 # 高速フーリエ変換におけるフレーム間のオーバーラップ長
    spec_frame_num = 64 # スペクトログラムのフレーム数 spec_freq_dim=512のとき、音声の長さが5秒の場合は128, 3秒の場合は64

    # 音声ファイルのパスを指定
    # target_voice_file = "./data/voice100_noise10/test/BASIC5000_0853_target.wav"
    # mixed_audio_file = "./data/voice100_noise10/test/BASIC5000_0853_001_mixed.wav"
    # target_voice_file = "./data/voice100_noise200/test/BASIC5000_1734_target.wav"
    # interference_audio_file = "./data/voice100_noise200/test/BASIC5000_1734_002_noise.wav"
    # mixed_audio_file = "./data/voice100_noise200/test/BASIC5000_1734_002_mixed.wav"

    # target_voice_file = "./data/voice100_noise100_3sec/test/BASIC5000_2009_target.wav"
    # interference_audio_file = "./data/voice100_noise100_3sec/test/BASIC5000_2009_008_noise.wav"
    # mixed_audio_file = "./data/voice100_noise100_3sec/test/BASIC5000_2009_008_mixed.wav"

    # target_voice_file = "../AudioDatasets/NoisySpeechdatabase/clean_testset_wav_16kHz/p232_013.wav"
    # interference_audio_file = "../AudioDatasets/NoisySpeechdatabase/interference_testset_wav_16kHz/p232_013.wav"
    # mixed_audio_file = "../AudioDatasets/NoisySpeechdatabase/noisy_testset_wav_16kHz/p232_013.wav"

    target_voice_file = "../AudioDatasets/NoisySpeechDatabase/clean_testset_wav_16kHz/p257_426.wav"
    interference_audio_file = "../AudioDatasets/NoisySpeechDatabase/interference_testset_wav_16kHz/p257_426.wav"
    mixed_audio_file = "../AudioDatasets/NoisySpeechDatabase/noisy_testset_wav_16kHz/p257_426.wav"

    wave_dir = "./output/wave/"
    os.makedirs(wave_dir, exist_ok=True)
    # オーディオファイルに対応する音声の波形を保存
    wave_image_dir = "./output/wave_image/"
    os.makedirs(wave_image_dir, exist_ok=True)
    # オーディオファイルに対応するスペクトログラムを保存
    spec_dir = "./output/spectrogram/"
    os.makedirs(spec_dir, exist_ok=True)

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    # checkpoint_path = "./ckpt/ckpt_voice100_noise200_0806/ckpt_epoch200.pt"
    # checkpoint_path = "./ckpt/ckpt_voice100_noise100_3sec_0819/ckpt_epoch30.pt"
    checkpoint_path = "./ckpt/ckpt_NoisySpeechDataset_fft_512_kernel3_0923/ckpt_epoch280.pt"
    # ネットワークモデルを指定
    model = Unet_kernel3()
    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 学習済みのパラメータをロード
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    # print("モデルのパラメータ数：", count_parameters(model))
    # Unetを使って推論
    # ネットワークを推論モードへ
    model.eval()

    # 分離処理の開始時間
    start_time = time.perf_counter()
    # 音声データをロード(現在は学習時と同じ処理をしているが、いずれはマイクロホンのリアルストリーミング音声を入力にしたい)
    mixed_audio_data = load_audio_file(mixed_audio_file, audio_length, sampling_rate)
    #　音声データをスペクトログラムに変換
    mixed_mag, mixed_phase = wave_to_spec(mixed_audio_data, fft_size, hop_length) # wavをスペクトログラムへ
    # スペクトログラムを正規化
    max_mag = mixed_mag.max()
    normed_mixed_mag = mixed_mag / max_mag
    # データの形式をモデルに入力できる形式に変更する
    # # librosaを使って、スペクトログラムを算出すると周波数要素が513になるので、512にスライス
    # mag_sliced = normed_mixed_mag[:int(fft_size/2), :]
    # phase_sliced = mixed_phase[:int(fft_size/2), :]
    # # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
    # mag_padded = np.pad(mag_sliced, [(0, 0), (0, spec_frame_num - mixed_mag.shape[1])], 'constant')
    # phase_padded = np.pad(phase_sliced, [(0, 0), (0, spec_frame_num - mixed_mag.shape[1])], 'constant')
    # カーネル3×3版
    mag_padded = np.pad(normed_mixed_mag, [(0, 0), (0, 513-normed_mixed_mag.shape[1])], 'constant')
    """mag_padded: (257, 513)"""
    phase_padded = np.pad(mixed_phase, [(0, 0), (0, 513-mixed_phase.shape[1])], 'constant')
    """phase_padded: (257, 513)"""
    # 0次元目と1次元目に次元を追加
    mag_expanded = mag_padded[np.newaxis, np.newaxis, :, :]
    """mag_padded: (1, 1, 257, 513)"""
    phase_expanded = phase_padded[np.newaxis, np.newaxis, :, :]
    """phase_expanded: (1, 1, 257, 513)"""
    # numpy形式のデータをpytorchのテンソルに変換
    mag_tensor = torch.from_numpy(mag_expanded.astype(np.float32)).clone()
    # 環境音のmaskを計算
    mask = model(mag_tensor)
    # pytorchのtensorをnumpy配列に変換
    mask = mask.to(device).detach().numpy().copy()
    # 人の声を取り出す
    normed_separated_voice_mag = mask * mag_expanded
    # 正規化によって小さくなった音量を元に戻す
    separated_voice_mag = normed_separated_voice_mag * max_mag
    # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
    voice_spec = separated_voice_mag * phase_expanded
    """voice_spec: (batch_size=1, channels=1, freq_bins=257, time_steps=513)"""
    voice_spec = np.squeeze(voice_spec)
    """voice_spec: (freq_bins=257, time_steps=513)"""
    # paddingした分を元に戻す
    voice_spec = voice_spec[:, :normed_mixed_mag.shape[1]]
    """voice_spec: (freq_bins=257, time_steps=301)"""
    estimated_voice_data = spec_to_wav(voice_spec, hop_length)
    # オーディオデータを保存
    estimated_voice_path = os.path.join(wave_dir, "estimated_voice.wav")
    save_audio_file(estimated_voice_path, estimated_voice_data, sampling_rate=16000)
    # 分離処理の終了時間
    finish_time = time.perf_counter()
    # 処理時間
    process_time = finish_time - start_time
    print("処理時間：", str(process_time) + 'sec')
    # デバッグ用に元のオーディオファイルとそのスペクトログラムを保存
    # オリジナル音声
    target_voice_path = os.path.join(wave_dir, "target_voice.wav")
    target_voice_data = load_audio_file(target_voice_file, audio_length, sampling_rate)
    save_audio_file(target_voice_path, target_voice_data, sampling_rate=16000)
    # 外的雑音
    interference_audio_path = os.path.join(wave_dir, "interference_audio.wav")
    interference_audio_data = load_audio_file(interference_audio_file, audio_length, sampling_rate)
    save_audio_file(interference_audio_path, interference_audio_data, sampling_rate=16000)
    # 混合音声
    mixed_audio_path = os.path.join(wave_dir, "mixed_audio.wav")
    save_audio_file(mixed_audio_path, mixed_audio_data, sampling_rate=16000)

    # 音声の波形を画像として保存
    # オリジナル音声の波形
    target_voice_img_path = os.path.join(wave_image_dir, "target_voice.png")
    wave_plot(target_voice_path, target_voice_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)
    # 外的雑音の波形
    interference_img_path = os.path.join(wave_image_dir, "interference_audio.png")
    wave_plot(interference_audio_path, interference_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)
    # 分離音の波形
    estimated_voice_img_path = os.path.join(wave_image_dir, "estimated_voice.png")
    wave_plot(estimated_voice_path, estimated_voice_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)
    # 混合音声の波形
    mixed_audio_img_path = os.path.join(wave_image_dir, "mixed_audio.png")
    wave_plot(mixed_audio_path, mixed_audio_img_path, audio_length, ylim_min=-1.0, ylim_max=1.0)

    # スペクトログラムを画像として保存
    # 現在のディレクトリ位置を取得
    base_dir = os.getcwd()
    # オリジナル音声のスペクトログラム
    target_voice_spec_path = os.path.join(spec_dir, "target_voice.png")
    spec_plot(base_dir, target_voice_path, target_voice_spec_path, audio_length)
    # 外的雑音のスペクトログラム
    interference_audio_spec_path = os.path.join(spec_dir, "interference_audio.png")
    spec_plot(base_dir, interference_audio_path, interference_audio_spec_path, audio_length)
    # 分離音のスペクトログラム
    estimated_voice_spec_path = os.path.join(spec_dir, "estimated_voice.png")
    spec_plot(base_dir, estimated_voice_path, estimated_voice_spec_path, audio_length)
    # 混合音声のスペクトログラム
    mixed_audio_spec_path = os.path.join(spec_dir, "mixed_audio.png")
    spec_plot(base_dir, mixed_audio_path, mixed_audio_spec_path, audio_length)

    # 音声評価
    sdr_mix, sir_mix, sar_mix, sdr_est, sir_est, sar_est = audio_eval(audio_length, target_voice_path, interference_audio_path, mixed_audio_path, estimated_voice_path)
    print("SDR_mix: {:.3f}, SIR_mix: {:.3f}, SAR_mix: {:.3f}".format(sdr_mix, sir_mix, sar_mix))
    print("SDR_est: {:.3f}, SIR_est: {:.3f}, SAR_est: {:.3f}".format(sdr_est, sir_est, sar_est))
