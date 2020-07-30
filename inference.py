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


from train import Unet
from mk_data import load_audio_file, save_audio_file, wave_to_spec


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
def spec_plot(base_dir, wav_path, save_path):
    # soxコマンドによりwavファイルからスペクトログラムの画像を生成
    cmd1 = "sox {} -n trim 0 5 rate 16.0k spectrogram".format(wav_path)
    subprocess.call(cmd1, shell=True)
    # 生成されたスペクトログラム画像を移動
    #(inference.pyを実行したディレクトリにスペクトログラムが生成されてしまうため)
    spec_path = os.path.join(base_dir, "spectrogram.png")
    cmd2 = "mv {} {}".format(spec_path, save_path)
    subprocess.call(cmd2, shell=True)

if __name__ == '__main__':

    sampling_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 5 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適かも？
    fft_size = 1024 # 高速フーリエ変換のフレームサイズ
    hop_length = 768 # 高速フーリエ変換におけるフレーム間のオーバーラップ長

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    checkpoint_path = "./ckpt/ckpt_voice100_noise200_0729/ckpt_epoch300.pt"
    # ネットワークモデルを指定
    model = Unet()
    # GPUが使える場合はGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 学習済みのパラメータをロード
    model_params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_params['model_state_dict'])
    # Unetを使って推論
    # ネットワークを推論モードへ
    model.eval()

    # 分離処理の開始時間
    start_time = time.time()
    # 音声ファイルのパスを指定
    target_voice_file = "./data/voice100_noise10/test/BASIC5000_0853_target.wav"
    mixed_audio_file = "./data/voice100_noise10/test/BASIC5000_0853_001_mixed.wav"
    # original_audio_file = "./data/voice100_noise10/test/BASIC5000_1222_target.wav"
    # mixed_audio_file = "./data/voice100_noise10/test/BASIC5000_1222_007_mixed.wav"
    # 音声データをロード(現在は学習時と同じ処理をしているが、いずれはマイクロホンのリアルストリーミング音声を入力にしたい)
    mixed_audio_data = load_audio_file(mixed_audio_file, audio_length, sampling_rate)
    #　音声データをスペクトログラムに変換
    mixed_mag, mixed_phase = wave_to_spec(mixed_audio_data, fft_size, hop_length) # wavをスペクトログラムへ
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
    mask = model(mag_tensor)
    # pytorchのtensorをnumpy配列に変換
    mask = mask.detach().numpy()
    # 人の声を取り出す
    normed_separated_voice_mag = mask * mag_expanded
    # 正規化によって小さくなった音量を元に戻す
    separated_voice_mag = normed_separated_voice_mag * max_mag
    # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
    voice_spec = separated_voice_mag * phase_expanded  # shape:(1, 1, 512, 128)
    voice_spec = np.squeeze(voice_spec) # shape:(512, 128)
    predicted_voice_data = spec_to_wav(voice_spec, hop_length)
    # オーディオファイルとそのスペクトログラムを保存
    predicted_voice_path = "./output/wav/predicted_voice.wav"
    save_audio_file(predicted_voice_path, predicted_voice_data, sampling_rate=16000)
    # 分離処理の終了時間
    finish_time = time.time()
    # 処理時間
    process_time = finish_time - start_time
    print("処理時間：", str(process_time) + 'sec')

    # デバッグ用に元のオーディオファイルとそのスペクトログラムを保存
    # オリジナル音声
    target_voice_path = "./output/wav/target_voice.wav"
    # cmd = "cp {} {}".format(original_audio_file, original_voice_path)
    # subprocess.call(cmd, shell=True)
    target_voice_data = load_audio_file(target_voice_file, audio_length, sampling_rate)
    save_audio_file(target_voice_path, target_voice_data, sampling_rate=16000)
    # 混合音声
    mixed_audio_path = "./output/wav/mixed_audio.wav"
    save_audio_file(mixed_audio_path, mixed_audio_data, sampling_rate=16000)

    # オーディオファイルに対応するスペクトログラムを保存
    # 現在のディレクトリ位置を取得
    base_dir = os.getcwd()
    # オリジナル音声のスペクトログラム
    target_voice_spec_path = "./output/spectrogram/target_voice.png"
    spec_plot(base_dir, target_voice_file, target_voice_spec_path)
    # 分離音のスペクトログラム
    predicted_voice_spec_path = "./output/spectrogram/predicted_voice.png"
    spec_plot(base_dir, predicted_voice_path, predicted_voice_spec_path)
    # 混合音声のスペクトログラム
    mixed_audio_spec_path = "./output/spectrogram/mixed_audio.png"
    spec_plot(base_dir, mixed_audio_path, mixed_audio_spec_path)
