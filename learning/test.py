import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 2. ネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, num_output_classes=10):
        super(Net, self).__init__()
        ## 入力はRGB画像 (チャネル数=3)
        ## 出力が8チャネルとなるような畳み込みを行う
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size = 3, padding=1)
        
        ## 画像を32x32から16x16に縮小する
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        ## 8ch -> 16ch, 16x16 -> 8x8
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size = 3, padding=1)
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

        ### ドロップアウト
        #x = self.dropout1(x)
        ## フォーマット変換
        x = x.view(-1, 50 * 8 * 8)

        ## 全結合層
        x = self.fc1(x)
        x = F.relu(x)

        ### ドロップアウト
        #x = self.dropout2(x)
        x = self.fc2(x)

        return x

# 1. データセットの読み出し
## CIFAR-10 の学習・テストデータの取得
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

## データの読み出し方法の定義
## 1stepの学習・テストごとに4枚ずつ画像を読みだす
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# 使用デバイス自動定義(CUDAが使えればCUDA、そうでないときはCPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

net = Net()
net = net.to(device)

## ロス関数、最適化器の定義
loss_func = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# 3. 学習
## データセット内の全画像を2回使用するまでループ
for epoch in range(2):
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
            running_loss = 0.0

# 4. テスト
ans = []
pred = []
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs)
    outputs = outputs.to('cpu')
    ans += labels.tolist()
    pred += torch.argmax(outputs, 1).tolist()

print(confusion_matrix(ans, pred))
print(accuracy_score(ans, pred))

# 5. モデルの保存
torch.save(net.state_dict(), 'model.pt')
