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
import librosa.display
import time
import pandas as pd
import glob
import pickle as pl

# データの前処理を行うクラス
class NumpyToTensor():
    def __init__(self, spec_freq_dim, spec_frame_num):
        self.spec_freq_dim = spec_freq_dim
        self.spec_frame_num = spec_frame_num

    def __call__(self, data_path):
        # .npyファイルからnumpy形式のデータを読み込む
        load_data = np.load(data_path)
        # モデルの入力形式に合わせて、データ0埋めと次元調整を行う
        # librosaを使って、スペクトログラムを算出すると周波数要素が513になるので、512にスライス
        load_data_sliced = load_data[(load_data.shape[0]-self.spec_freq_dim):, :]
        # モデルの入力サイズに合わせて、スペクトログラムの後ろの部分を0埋め(パディング)
        load_data_padded = np.pad(load_data_sliced, [(0, 0), (0, self.spec_frame_num-load_data_sliced.shape[1])], 'constant')
        # 0次元目に次元を追加
        load_data_expanded = np.expand_dims(load_data_padded, 0)
        # numpy形式のデータをpytorchのテンソルに変換
        tensor_data = torch.from_numpy(load_data_expanded)
        return tensor_data

# Unetのモデルを定義
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # encoderの層
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1) # ２次元データの畳み込み演算を行う層　引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
        self.norm1 = nn.BatchNorm2d(16) # batch normalizationを行う層　引数は入力データのチャンネル数
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.norm6 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # decoderの層
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.denorm1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.denorm2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.denorm3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1)
        self.denorm4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1)
        self.denorm5 = nn.BatchNorm2d(16)
        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = torch.randn(64, 1, 512, 128) # (batch_size, num_channels, height, width)
        h1 = self.leaky_relu(self.norm1(self.conv1(x)))
        h2 = self.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = self.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = self.leaky_relu(self.norm4(self.conv4(h3)))
        h5 = self.leaky_relu(self.norm5(self.conv5(h4)))
        h6 = self.leaky_relu(self.norm6(self.conv6(h5)))
        dh1 = self.relu(self.dropout(self.denorm1(self.deconv1(h6))))
        dh2 = self.relu(self.dropout(self.denorm2(self.deconv2(torch.cat((dh1, h5), dim=1)))))
        dh3 = self.relu(self.dropout(self.denorm3(self.deconv3(torch.cat((dh2, h4), dim=1)))))
        dh4 = self.relu(self.denorm4(self.deconv4(torch.cat((dh3, h3), dim=1))))
        dh5 = self.relu(self.denorm5(self.deconv5(torch.cat((dh4, h2), dim=1))))
        dh6 = self.sigmoid(self.deconv6(torch.cat((dh5, h1), dim=1)))
        return dh6

# trainデータとvalidationデータのファイルパスリストを取得
def mk_datapath_list(dataset_dir):
    # trainデータのパスとラベルを取得
    train_mixed_path_template = os.path.join(dataset_dir, "train/*_mixed.npy")
    train_mixed_spec_list = glob.glob(train_mixed_path_template) # 混合音声のスペクトログラムのパスリスト
    train_target_spec_list = [] # 声だけのスペクトログラム(正解ラベル)のパスリスト
    for train_mixed_spec_path in train_mixed_spec_list:
        train_mixed_file_name = train_mixed_spec_path.split('/')[-1] # (例)BASIC5000_0001_001_mixed.npy
        train_target_file_name = train_mixed_file_name.rsplit('_', maxsplit=2)[0] + "_target.npy" # (例)BASIC5000_0001_target.npy
        train_target_file_path = os.path.join(dataset_dir, "train/{}".format(train_target_file_name))
        train_target_spec_list.append(train_target_file_path)

    # valデータのパスとラベルを取得
    val_mixed_path_template = os.path.join(dataset_dir, "val/*_mixed.npy")
    val_mixed_spec_list = glob.glob(val_mixed_path_template) # 混合音声のスペクトログラムのパスリスト
    val_target_spec_list = [] # 声だけのスペクトログラム(正解ラベル)のパスリスト
    for val_mixed_spec_path in val_mixed_spec_list:
        val_mixed_file_name = val_mixed_spec_path.split('/')[-1] # (例)BASIC5000_0001_001_mixed.npy
        val_target_file_name = val_mixed_file_name.rsplit('_', maxsplit=2)[0] + "_target.npy" # (例)BASIC5000_0001_target.npy
        val_target_file_path = os.path.join(dataset_dir, "val/{}".format(val_target_file_name))
        val_target_spec_list.append(val_target_file_path)

    return train_mixed_spec_list, train_target_spec_list, val_mixed_spec_list, val_target_spec_list


# データセットのクラス
class VoiceDataset(data.Dataset):
    def __init__(self, mixed_spec_list, target_spec_list, transform):
        self.mixed_spec_list = mixed_spec_list # 混合音声のスペクトログラムのファイルリスト
        self.target_spec_list = target_spec_list # 声だけのスペクトログラムのファイルリスト
        self.transform = transform

    def __len__(self):
        return len(self.mixed_spec_list)

    def __getitem__(self, index):
        # スペクトログラムのファイルパスを取得
        mixed_spec_path = self.mixed_spec_list[index]
        target_spec_path = self.target_spec_list[index]
        # numpy形式のスペクトログラムを読み込み、pytorchのテンソルに変換
        mixed_spec = self.transform(mixed_spec_path)
        target_spec = self.transform(target_spec_path)

        return mixed_spec, target_spec


# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir):

    # GPUが使える場合あはGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 各カウンタを初期化
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    # 学習済みモデルのパラメータを保存するディレクトリを作成
    os.makedirs(param_save_dir, exist_ok=True)

    # epochごとのループ
    for epoch in range(num_epochs):

        # 開始時刻を記録
        epoch_start_time = time.time()
        iter_start_time = time.time()

        print("エポック {}/{}".format(epoch+1, num_epochs))

        # モデルのモードを切り替える(学習 ⇔ 検証)
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # 学習モード
            else:
                if (epoch % 10 == 0): # 10回ごとに検証
                    net.eval() # 検証モード
                else:
                    continue

            # データローダからミニバッチずつ取り出すループ
            for mixed_spec, target_spec in dataloaders_dict[phase]:
                # GPUが使える場合、データをGPUへ送る
                mixed_spec = mixed_spec.to(device)
                target_spec = target_spec.to(device)
                # optimizerを初期化
                optimizer.zero_grad()
                # 順伝播
                with torch.set_grad_enabled(phase == 'train'):
                    # 混合音声のスペクトログラムをモデルに入力し、ノイズマスクを算出
                    mask = net(mixed_spec)
                    # 損失を計算
                    loss = criterion(mask*mixed_spec, target_spec)
                    # 学習時は誤差逆伝播(バックプロパゲーション)
                    if phase == 'train':
                        loss.backward()
                        # パラメータ更新
                        optimizer.step()
                        # 10iterationごとにlossと処理時間を表示
                        if (iteration % 10 == 0):
                            iter_finish_time = time.time()
                            duration_per_ten_iter = iter_finish_time - iter_start_time
                            # 0次元のテンソルから値を取り出す場合は「.item()」を使う
                            print("イテレーション {} | Loss:{:.4f} | 経過時間:{:.4f}[sec]".format(iteration, loss.item(), duration_per_ten_iter))
                            epoch_train_loss += loss.item()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        # epochごとのlossと正解率を表示
        epoch_finish_time = time.time()
        duration_per_epoch = epoch_finish_time - epoch_start_time
        print("=" * 30)
        print("エポック {} | Epoch train Loss:{:.4f} | Epoch val Loss:{:.4f}".format(epoch+1, epoch_train_loss, epoch_val_loss))
        print("経過時間:{:.4f}[sec/epoch]".format(duration_per_epoch))

        # 学習経過を分析できるようにcsvファイルにログを保存 → tensorboardに変更しても良いかも
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log.csv")

        # epochごとの損失を初期化
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # 学習したモデルのパラメータを保存
        if ((epoch+1) % 10 == 0):
            param_save_path = os.path.join(param_save_dir, "ckpt_epoch{}.pt".format(epoch+1))
            torch.save(net.state_dict(), param_save_path)


if __name__ == '__main__':
    # 各パラメータを設定
    batch_size = 64 # バッチサイズ
    spec_freq_dim = 512 # スペクトログラムの周波数次元数
    spec_frame_num = 128 # スペクトログラムのフレーム数
    # モデルを作成
    net = Unet()
    # データセットを作成
    dataset_dir = "./data/voice1000_noise100/"
    train_mixed_spec_list, train_target_spec_list, val_mixed_spec_list, val_target_spec_list = mk_datapath_list(dataset_dir)
    # 前処理クラスのインスタンスを作成
    transform = NumpyToTensor(spec_freq_dim, spec_frame_num) # numpy形式のスペクトログラムをpytorchのテンソルに変換する
    # データセットのインスタンスを作成
    train_dataset = VoiceDataset(train_mixed_spec_list, train_target_spec_list, transform=transform)
    val_dataset = VoiceDataset(val_mixed_spec_list, val_target_spec_list, transform=transform)
    #　データローダーを作成
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader} # データローダーを格納するリスト
    # 損失関数を定義
    criterion = nn.L1Loss(reduction='sum') # L1Loss(input, target) : inputとtargetの各要素の差の絶対値の和
    # 最適化手法を定義
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 各種設定 → いずれ１つのファイルからデータを読み込ませたい
    num_epochs = 50 # epoch数を指定
    checkpoint_dir = "./ckpt" # 学習済みモデルを保存するディレクトリのパスを指定
    #　モデルを学習
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs, checkpoint_dir)
