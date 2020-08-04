
# ACRi ルームのサーバー上での動かし方

ACRi ルームの Alveo サーバー上で本リポジトリで作成する MNIST モデルを実行する方法を記します。
ターゲットとなる環境は、`as001` サーバーです。

ACRi ルームのサーバー上では外部ネットワークへの接続が不可なため、まずは自前の開発マシンでの準備が必要です。


## 自前の開発マシンでの作業手順

1. このリポジトリをclone 
2. `thirdparty` 以下の`download.sh` を実行
3. `learning` 以下に入り学習を行う (手順は[README.md](../README.md) 内に記載)
4. リポジトリのコード全体を圧縮し、ACRi ルームのサーバーのホームディレクトリ上にコピーする

## ACRi ルームのサーバー上での作業手順

公式の利用方法を元にログインします。  
- サーバ全般: http://gw.acri.c.titech.ac.jp/wp/manual/how-to-reserve
- Alveoサーバ: http://gw.acri.c.titech.ac.jp/wp/manual/alveo-server

レポート表示時に GUI 機能を使うため、リモートデスクトップの使用を推奨します。

### 準備

`/home/<username>/dnn-kernel-fpga.zip` に圧縮済みのコードがある前提で説明します。

まず、高速なローカルディレクトリである`/scratch` 上にデータをコピーします。  
```
$ cp /home/<username>/dnn-kernel-fpga.zip /scratch
```

ワーキングディレクトリを`/scratch`に移動しコピーしたファイルを解凍します。  
```
$ cd /scratch
$ unzip dnn-kernel-fpga.zip
```

cmake 3.16.8 にパスを通します。  
```
$ export PATH=/scratch/dnn-kernel-fpga/thirdparty/cmake-3.16.8-Linux-x86_64/bin:${PATH}`
```

以降は、[README.md](../README.md) に記載のビルド・推論処理の手順を行います。  
MNIST の学習、テストの手順は ACRi ルームのサーバー上では行えません。
