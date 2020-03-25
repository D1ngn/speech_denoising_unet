# 必要モジュールのimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wave


from train import Unet
from mk_data import load_audio_file, save_audio_file, wav_to_spec


# スペクトログラムを音声データに変換する
def spec_to_wav(spec, hop_length):
    # 逆短時間フーリエ変換(iSTFT)を行い、スペクトログラムから音声データを取得
    wav_data = librosa.istft(spec, hop_length=hop_length)
    return wav_data


# スペクトログラムを図にプロットする関数
def spec_plot(input_spec, save_path):
    # パワースペクトルを対数パワースペクトルに変換
    log_power_spec = librosa.amplitude_to_db(input_spec, ref=np.max)
    plt.figure(figsize=(12,5)) # 図の大きさを指定
    librosa.display.specshow(log_power_spec, x_axis='time', y_axis='log')
    plt.title('Spectroram') # タイトル
    plt.xlabel("time[s]") # 横軸のラベル
    plt.ylabel("amplitude") # 縦軸のラベル
    plt.colorbar(format='%+02.0f dB') # カラーバー表示
    plt.savefig(save_path)


if __name__ == '__main__':

    # 学習済みのパラメータを保存したチェックポイントファイルのパスを指定
    checkpoint_path = "./ckpt/ckpt_epoch50.pt"
    # ネットワークモデルを指定
    net = Unet()
    # GPUが使える場合あはGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)
    # 学習済みのパラメータをロード
    net_params = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(net_params['model_state_dict'])
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
    input_mag, input_phase = wav_to_spec(input_audio_data, fft_size, hop_length) # wavをスペクトログラムへ
    # スペクトログラムを正規化
    max_mag = input_mag.max()
    normed_input_mag = input_mag / max_mag
    # データの形式をモデルに入力できる形式に変更する
    # librosaを使って、スペクトログラムを算出すると周波数要素が513になるので、512にスライス
    mag_sliced = normed_input_mag[1:, :]
    phase_sliced = input_phase[1:, :]
    # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
    mag_padded = np.pad(mag_sliced, [(0, 0), (0, 128 - mag_sliced.shape[1])], 'constant')
    phase_padded = np.pad(phase_sliced, [(0, 0), (0, 128 - mag_sliced.shape[1])], 'constant')
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
    separated_voice_mag = mask * mag_expanded
    # マスクした後の振幅スペクトログラムに入力音声の位相スペクトログラムを掛け合わせて音声を復元
    voice_spec = separated_voice_mag * phase_expanded  # shape:(1, 1, 512, 128)
    voice_spec = np.squeeze(voice_spec) # shape:(512, 128)
    masked_voice_data = spec_to_wav(voice_spec, hop_length)

    # オーディオファイルとそのスペクトログラムを保存
    masked_voice_path = "./output/wav/masked_voice.wav"
    save_audio_file(masked_voice_path, masked_voice_data, sampling_rate=16000)

    # デバッグ用に元のオーディオファイルとそのスペクトログラムを保存
    mixed_voice_path = "./output/wav/mixed_voice.wav"
    save_audio_file(mixed_voice_path, input_audio_data, sampling_rate=16000)

    # オーディオファイルに対応するスペクトログラムを保存
    # 分離音のスペクトログラム
    masked_voice_spec_path = "./output/spectrogram/masked_voice.png"
    separated_voice_mag = np.squeeze(separated_voice_mag) # 次元削減
    spec_plot(separated_voice_mag, masked_voice_spec_path)
    # 混合音声のスペクトログラム
    mixed_voice_spec_path = "./output/spectrogram/mixed_voice.png"
    spec_plot(input_mag, mixed_voice_spec_path)
