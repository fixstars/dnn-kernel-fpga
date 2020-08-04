
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

ワーキングディレクトリを移動し解凍します。  
```
$ cd /scratch
$ unzip dnn-kernel-fpga.zip
```

cmake 3.16 にパスを通します。  
```
$ export PATH=/scratch/dnn-kernel-fpga/thirdparty/cmake-3.16.8-Linux-x86_64/bin:${PATH}`
```

cmake プロジェクトを作成し、ホストアプリケーション等をビルドします。  
```
$ mkdir /scratch/dnn-kernel-fpga/build
$ cd /scratch/dnn-kernel-fpga/build
$ cmake .. -DTARGET_BOARD=u200
$ make
```

### ビットストリームの合成

この手順は2時間程度かかります。  

環境変数を設定します。  
```
$ source /tools/Xilinx/Vitis/2019.2/settings64.sh
$ source /opt/xilinx/xrt/setup.sh
```

.xo 及び .xclbin の作成を行います。  
```
$ cd /scratch/dnn-kernel-fpga/build
$ cmake --build . --target inference_top_hw_xo
$ cmake --build . --target inference_top_hw
```

### 実行

`xrt.ini` を準備します。  
```
$ cd /scratch/dnn-kernel-fpga/build
$ echo "[Debug]\ntimeline_trace=true" > xrt.ini
```

アプリケーションを実行します。  
```
$ ./host/run_inference host/inference_top_hw.xclbin inference_top
```

1分程度で実行が完了し、次のようなログが出ます。  
```
Elapsed time: 18.6115 [ms/image]
accuracy: 0.976 
```

その後、Vitis Analyzer によりレポートが確認できます。  
```
$ vitis_analyzer inference_top_hw.xclbin.run_summary &
```

