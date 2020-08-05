# DNN-Kernel-FPGA

Deep Learning の FPGA 向けフルスクラッチ実装

## 概要

このプロジェクトは、小規模な畳み込みネットワークを FPGA で実装したものです。  
MNIST データセットをターゲットに、フルスクラッチで書いたネットワークモデルを Alveo FPGA カード上で動作させます。

特にACRi ルーム上でこのコードを使用する場合は、[doc/acri-room-howto.md](doc/acri-room-howto.md) を参考にしてください。

## 開発環境
- Ubuntu (>= 18.04)
- Python (>= 3.5.2)
- CMake (>= 3.11)
- Vivado HLS (>= 2019.2)

## MNIST の学習

以下はvirtualenv を使用しているので、その他の python 仮想環境を使用する場合は各々変更してください。

```sh
cd learning
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python train_mnist.py
```

## ビルド

#### ホストアプリケーションなど
```sh
mkdir build && cd build
cmake -DTARGET_BOARD=u200 ../
cmake --build .
```

#### FPGA イメージ

環境変数を設定します。  
```
$ source /tools/Xilinx/Vitis/2019.2/settings64.sh
$ source /opt/xilinx/xrt/setup.sh
```

ビットストリームの合成を行います。  
この手順は2時間程度かかります。  
```sh
cmake --build . --target inference_top_hw_xo
cmake --build . --target inference_top_hw
```

合成レポートは次のように確認できます。  
```sh
vitis_analyzer host/inference_top_hw.xclbin.link_summary
```

## 推論処理

### 推論の実行

トレース取得用に `xrt.ini` を作成します。
```sh
echo -e "[Debug]\ntimeline_trace=true" > xrt.ini
```

以下のコマンドで推論処理が実行されます。
```sh
./host/run_inference ./host/inference_top_hw.xclbin inference_top
```

実行レポートは次のように確認できます。  
```sh
vitis_analyzer inference_top_hw.xclbin.run_summary
```

## テスト

#### 単体テスト

以下のようにして単体テストが可能です (ReLU の場合) 。  

```sh
ctest -V -R "relu_ref"         # Test of reference implementation
ctest -V -R "relu_hls_csim"    # C simulation test of HLS implementation
ctest -V -R "relu_hls_cosim"   # C/RTL co-simulation test of HLS implementation
```

