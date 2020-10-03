import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import glob
import random

from tqdm import tqdm


#　音声データをロードし、指定された秒数とサンプリングレートでリサンプル
def load_audio_file(file_path, length, sampling_rate=16000):
    data, sr = librosa.load(file_path, sr=sampling_rate)
    # data, sr = sf.read(file_path)
    # データが設定値よりも大きい場合は大きさを超えた分をカットする
    # データが設定値よりも小さい場合はデータの後ろを0でパディングする
    if len(data) > sampling_rate*length:
        data = data[:sampling_rate*length]
    else:
        data = np.pad(data, (0, max(0, sampling_rate*length - len(data))), "constant")
    return data

# 音声データを指定したサンプリングレートで保存
def save_audio_file(file_path, data, sampling_rate=16000):
    # librosa.output.write_wav(file_path, data, sampling_rate) # 正常に動作しないので変更
    sf.write(file_path, data, sampling_rate)

# 2つのオーディオデータを足し合わせる
def audio_mixer(data1, data2):
    assert len(data1) == len(data2)
    mixed_audio = data1 + data2
    return mixed_audio

# 音声データをスペクトログラムに変換する
def wave_to_spec(data, fft_size, hop_length):
    # 短時間フーリエ変換(STFT)を行い、スペクトログラムを取得
    spec = librosa.stft(data, n_fft=fft_size, hop_length=hop_length)
    mag = np.abs(spec) # 振幅スペクトログラムを取得
    phase = np.exp(1.j * np.angle(spec)) # 位相スペクトログラムを取得(フェーザ表示)
    # mel_spec = librosa.feature.melspectrogram(data, sr=sr, n_mels=128) # メルスペクトログラムを用いる場合はこっちを使う
    return mag, phase

if __name__ == '__main__':
    # 各パラメータを設定
    sampling_rate = 16000 # 作成するオーディオファイルのサンプリング周波数を指定
    audio_length = 3 # 単位は秒(second) → fft_size=1024,hop_length=768のとき、audio_length=6が最適化かも？
    train_val_ratio = 0.9 # trainデータとvalidationデータの割合
    fft_size = 512 # 高速フーリエ変換のフレームサイズ
    hop_length = 160 # 高速フーリエ変換においてフレームをスライドさせる幅

    # 乱数を初期化
    random.seed(0)

    # データセットを格納するディレクトリを作成
    save_dataset_dir = "./data/NoisySpeechDataset_for_unet_fft_512/"
    os.makedirs(save_dataset_dir, exist_ok=True)

    # 人の発話音声のディレクトリを指定
    target_data_dir = "../AudioDatasets/NoisySpeechdatabase/clean_trainset_28spk_wav_16kHz/"
    # 外部雑音のディレクトリを指定
    interference_data_dir = "../AudioDatasets/NoisySpeechdatabase/interference_trainset_wav_16kHz/"
    # 混合音声のディレクトリを指定
    mixed_data_dir = "../AudioDatasets/NoisySpeechdatabase/noisy_trainset_28spk_wav_16kHz/"

    target_data_path_template = os.path.join(target_data_dir, "*.wav")
    target_list = glob.glob(target_data_path_template)
    # データセットをシャッフル
    random.shuffle(target_list)
    # データをtrainデータとvalidationデータに分割
    target_list_for_train = target_list[:int(len(target_list)*train_val_ratio)]
    target_list_for_val = target_list[int(len(target_list)*train_val_ratio):]
    print("オリジナルデータの数:", len(target_list))
    print("trainデータの数:", len(target_list_for_train))
    print("validationデータの数:", len(target_list_for_val))

    # trainデータを作成
    print("trainデータ作成中")
    train_data_path = os.path.join(save_dataset_dir, "train")
    os.makedirs(train_data_path, exist_ok=True)
    for target_path in tqdm(target_list_for_train):
        file_num = os.path.basename(target_path).split('.')[0] # (例)p226_001
        target_file_name = file_num + "_target.npy" # (例)p226_001_target.npy
        target_data = load_audio_file(target_path, audio_length, sampling_rate)
        # オーディオデータをスペクトログラムに変換
        target_mag, _ = wave_to_spec(target_data, fft_size, hop_length)
        # スペクトログラムを正規化
        max_mag = target_mag.max()
        normed_target_mag = target_mag / max_mag
        # .npy形式でスペクトログラムを保存
        target_save_path = os.path.join(train_data_path, target_file_name)
        np.save(target_save_path, normed_target_mag)
        # 混合音声をロード
        mixed_path = os.path.join(mixed_data_dir, file_num + ".wav")
        mixed_data = load_audio_file(mixed_path, audio_length, sampling_rate)
        # オーディオデータをスペクトログラムに変換
        mixed_mag, _ = wave_to_spec(mixed_data, fft_size, hop_length)
        normed_mixed_mag = mixed_mag / max_mag
        # .npy形式でスペクトログラムを保存
        mixed_file_name = file_num + "_mixed.npy" # (例)p226_001_mixed.npy
        mixed_save_path = os.path.join(train_data_path, mixed_file_name)
        np.save(mixed_save_path, normed_mixed_mag)

    # trainデータを作成
    print("validationデータ作成中")
    val_data_path = os.path.join(save_dataset_dir, "val")
    os.makedirs(val_data_path, exist_ok=True)
    for target_path in tqdm(target_list_for_val):
        file_num = os.path.basename(target_path).split('.')[0] # (例)p226_001
        target_file_name = file_num + "_target.npy" # (例)p226_001_target.npy
        target_data = load_audio_file(target_path, audio_length, sampling_rate)
        # オーディオデータをスペクトログラムに変換
        target_mag, _ = wave_to_spec(target_data, fft_size, hop_length)
        # スペクトログラムを正規化
        max_mag = target_mag.max()
        normed_target_mag = target_mag / max_mag
        # .npy形式でスペクトログラムを保存
        target_save_path = os.path.join(val_data_path, target_file_name)
        np.save(target_save_path, normed_target_mag)
        # 混合音声をロード
        mixed_path = os.path.join(mixed_data_dir, file_num + ".wav")
        mixed_data = load_audio_file(mixed_path, audio_length, sampling_rate)
        # オーディオデータをスペクトログラムに変換
        mixed_mag, _ = wave_to_spec(mixed_data, fft_size, hop_length)
        normed_mixed_mag = mixed_mag / max_mag
        # .npy形式でスペクトログラムを保存
        mixed_file_name = file_num + "_mixed.npy" # (例)p226_001_mixed.npy
        mixed_save_path = os.path.join(val_data_path, mixed_file_name)
        np.save(mixed_save_path, normed_mixed_mag)

    print("データ作成完了　保存先：{}".format(save_dataset_dir))
