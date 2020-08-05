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
import argparse
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

# Unetのモデルを定義
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # encoderの層
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), # ２次元データの畳み込み演算を行う層　引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
            nn.BatchNorm2d(16), # batch normalizationを行う層　引数は入力データのチャンネル数
            nn.LeakyReLU(0.2, inplace=True) # inplace=Trueにすることで、使用メモリを削減
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # decoderの層
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # ２次元データの畳み込み演算を行う層　引数は入力のチャンネル数、出力のチャンネル数、カーネル(フィルタ)の大きさ
            nn.BatchNorm2d(256), # batch normalizationを行う層　引数は入力データのチャンネル数
            nn.Dropout2d(p=0.5), # 50%の割合でドロップアウトを実行
            nn.ReLU(inplace=True) # inplace=Trueにすることで、使用メモリを削減
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x = torch.randn(64, 1, 512, 128) # (batch_size, num_channels, height, width)
        # encoder forward
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        h5 = self.conv5(h4)
        h6 = self.conv6(h5)
        # decoder forward
        dh1 = self.deconv1(h6)
        dh2 = self.deconv2(torch.cat((dh1, h5), dim=1))
        dh3 = self.deconv3(torch.cat((dh2, h4), dim=1))
        dh4 = self.deconv4(torch.cat((dh3, h3), dim=1))
        dh5 = self.deconv5(torch.cat((dh4, h2), dim=1))
        dh6 = self.deconv6(torch.cat((dh5, h1), dim=1))
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

# 既存のチェックポイントファイルをロード
def load_checkpoint(model, optimizer, checkpoint_path, device):
    # チェックポイントファイルがない場合エラー
    assert os.path.isfile(checkpoint_path)
    # チェックポイントファイルをロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_epoch = checkpoint['log_epoch']
    print("{}からデータをロードしました。エポック{}から学習を再開します。".format(checkpoint_path, start_epoch))
    return start_epoch, model, optimizer, log_epoch

# モデルを学習させる関数を作成
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir, checkpoint_path):

    # GPUが使える場合あはGPUを使用、使えない場合はCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：" , device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # 各カウンタを初期化
    start_epoch = 0
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    # 学習を再開する場合はパラメータをロード、最初から始める場合は特に処理は行われない
    if checkpoint_path is not None:
        start_epoch, model, optimizer, log_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
    else:
        print("checkpointファイルがありません。最初から学習を開始します。")

    # ネットワークをGPUへ
    model.to(device)

    # epochごとのループ
    for epoch in range(start_epoch, num_epochs):

        # 開始時刻を記録
        epoch_start_time = time.time()
        iter_start_time = time.time()

        print("エポック {}/{}".format(epoch+1, num_epochs))

        # モデルのモードを切り替える(学習 ⇔ 検証)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # 学習モード
            else:
                model.eval() # 検証モード
                # if ((epoch+1) % 10 == 0): # 10回ごとに検証
                #     model.eval() # 検証モード
                # else:
                #     continue

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
                    mask = model(mixed_spec)
                    # 損失を計算
                    loss = criterion(mask*mixed_spec, target_spec)
                    # 学習時は誤差逆伝播(バックプロパゲーション)
                    if phase == 'train':
                        # 誤差逆伝播を行い、勾配を算出
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
        log_save_path = os.path.join(param_save_dir, "log.csv")
        df.to_csv(log_save_path)

        # epochごとの損失を初期化
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # 学習したモデルのパラメータを保存
        if ((epoch+1) % 100 == 0):
            param_save_path = os.path.join(param_save_dir, "ckpt_epoch{}.pt".format(epoch+1))
            # torch.save(net.state_dict(), param_save_path) #　推論のみを行う場合
            # 学習を再開できるように変更
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'log_epoch': log_epoch
            }, param_save_path)


if __name__ == '__main__':

    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description='for unet train')
    parser.add_argument('--checkpoint_path', default=None, help="checkpoint path if you restart training")
    args = parser.parse_args()
    # 各パラメータを設定
    batch_size = 64 # バッチサイズ
    spec_freq_dim = 512 # スペクトログラムの周波数次元数
    spec_frame_num = 128 # スペクトログラムのフレーム数
    # モデルを作成
    model = Unet()
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
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader} # データローダーを格納するリスト
    # 損失関数を定義
    criterion = nn.L1Loss(reduction='sum') # L1Loss(input, target) : inputとtargetの各要素の差の絶対値の和
    # 最適化手法を定義
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 各種設定 → いずれ１つのファイルからデータを読み込ませたい
    num_epochs = 300 # epoch数を指定
    # 学習済みモデルのパラメータを保存するディレクトリを作成
    param_save_dir = "./ckpt" # 学習済みモデルのパラメータを保存するディレクトリのパスを指定
    os.makedirs(param_save_dir, exist_ok=True)
    #　モデルを学習
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, param_save_dir, checkpoint_path=args.checkpoint_path)
