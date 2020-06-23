import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import os

# 2. ネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, ksize = 3, pd = 1, drp = True, num_output_classes=10):
        super(Net, self).__init__()
        self.drp = drp
        ## 入力はRGB画像 (チャネル数=3)
        ## 出力が8チャネルとなるような畳み込みを行う
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size = ksize, padding=pd)
        ## 画像を32x32から16x16に縮小する
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        ## 8ch -> 16ch, 16x16 -> 8x8
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size = ksize, padding=pd)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        ## ドロップアウト
        self.dropout1 = torch.nn.Dropout2d(p=0.25)
        ## 全結合層
        ## 16chの8x8画像を1つのベクトルとみなし、要素数64のベクトルまで縮小
        self.fc1 = nn.Linear(50 * 8 * 8, 500)
        ## ドロップアウト
        self.dropout2 = torch.nn.Dropout2d(p=0.5)
        ## 全結合層その2
        ## 出力クラス数まで縮小
        self.fc2 = nn.Linear(500, num_output_classes)
        #self.cuda()
    def forward(self, x):
        ## 1層目の畳み込み
        ## 活性化関数 (activation) はReLU
        x = self.conv1(x)
        x = F.relu(x)
        ## 縮小
        x = self.pool1(x)
        ## 2層目+縮小
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        ## ドロップアウト
        if(self.drp == True):
            x = self.dropout1(x)
        ## フォーマット変換
        x = x.view(-1, 50 * 8 * 8)
        ## 全結合層
        x = self.fc1(x)
        x = F.relu(x)
        ## ドロップアウト
        if(self.drp == True):
            x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_and_test_main(f_ac, log_fname, ksize, pd, drp): ## 引数はファイル(名ではない), ログファイル名、学習データセット、テストデータセット、ドロップアウト層有無
    # 1. データセットの読み出し
    ## CIFAR-10 の学習・テストデータの取得
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    ## データの読み出し方法の定義
    ## 1stepの学習・テストごとに4枚ずつ画像を読みだす
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

    f_log = open(log_fname, "w")

    # 使用デバイス自動定義(CUDAが使えればCUDA、そうでないときはCPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {device}")
    net = Net(3, 1, drp, 10)
    net = net.to(device)
    ## ロス関数、最適化器の定義
    loss_func = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # 3. 学習
    ## データセット内の全画像を200回使用するまでループ
    for epoch in range(200):
        print(f"epoch={epoch}")
        f_log.write(f"epoch={epoch}\n")
        running_loss = 0
        ## データセット内でループ
        for i, data in enumerate(trainloader, 0):
            ## 入力バッチの読み込み (画像、正解ラベル)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ## 最適化器をゼロ初期化
            optimizer.zero_grad()
            ## 入力画像をモデルに通して出力ラベルを取得
            outputs = net(inputs)
            ## 正解との誤差の計算 + 誤差逆伝搬
            loss = loss_func(outputs, labels)
            loss.backward()
            ## 誤差を用いてモデルの最適化
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                msg = ('[%d, %5d] loss: %.3f\n' % (epoch + 1, i + 1, running_loss / 2000))
                f_log.write(msg)
                running_loss = 0.0
        # 4. テスト (各epochごと)
        ans = []
        pred = []
        for i, data in enumerate(testloader, 0):
        #for i, data in enumerate(trainloader, 0):
            inputs_test, labels_test = data
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(inputs_test)
            outputs_test = outputs_test.to('cpu')
            ans += labels_test.tolist()
            pred += torch.argmax(outputs_test, 1).tolist()
        #print(ans)
        #print(pred)
        print(f"{confusion_matrix(ans, pred)}")
        f_log.write(f"{confusion_matrix(ans, pred)}\n")
        print(accuracy_score(ans, pred))
        f_ac.write(f"{accuracy_score(ans, pred)}, ")
    f_ac.write("\n")
    f_log.close()
    # 5. モデルの保存
    torch.save(net.state_dict(), f"model_{ksize}_{drp}.pt")

def train_and_test(accuracy_fname):
    ## 0. ログ準備
    f_ac = open(accuracy_fname, "w")

    # 学習・推論本体
    ## カーネルサイズ3, ドロップアウトあり
    train_and_test_main(f_ac, "log_k3_drp.txt", 3, 1, True)
    ## カーネルサイズ3, ドロップアウトなし
    train_and_test_main(f_ac, "log_k3_nodrp.txt", 3, 1, False)
    ## カーネルサイズ5, ドロップアウトあり
    train_and_test_main(f_ac, "log_k5_drp.txt", 5, 2, True)
    ## カーネルサイズ5, ドロップアウトなし
    train_and_test_main(f_ac, "log_k5_nodrp.txt", 5, 2, False)

    ## ファイルを閉じる
    f_ac.close()

if __name__ == "__main__":
    train_and_test("accuracy.csv")
