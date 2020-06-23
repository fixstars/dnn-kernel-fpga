#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include <torch/torch.h>
namespace F = torch::nn::functional;
static const char* kObjectClasses[] = {
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"};
// ネットワーク定義
struct Net : torch::nn::Module {
    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 8, 3).padding(1)));
        pool1 = register_module("pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).padding(1)));
        pool2 = register_module("pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        fc1 = register_module("fc1", torch::nn::Linear(16 * 8 * 8, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 10));
    };
    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = torch::relu(x);
        x = pool1->forward(x);
        x = conv2->forward(x);
        x = torch::relu(x);
        x = pool2->forward(x);
        // フォーマット変換
        x = x.reshape({x.size(0), {16 * 8 * 8}});
        x = fc1->forward(x);
        x = torch::relu(x);
        x = fc2->forward(x);
        return x;
    }
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
/*
class Cifar10 { // 1.まずこの部分はコメントアウトして、モデルのコンパイルを通す 2.クラスが
public:
    Cifar10(const std::string& data_path) { // バイナリバージョンを読む想定　
        std::vector<uint8_t> raw;
        // read data from file
        std::ifstream ifs(data_path);
        ifs.seek(0, std::ifstream::end);
        raw.resize(ifs.tellg());
        ifs.seek(0, std::ifstream::beg);
        ifs.read(reinterpret_cast<char*>(raw.data()), raw.size());
        std::size_t nimages = raw.size() / 3073;
        data_ = Torch::empty({nimages, 3, 32, 32});
        labels_ = Torch::empty({nimages, 1});
        for (std::size_t i = 0; i < nimages.size(); i++) {
            labels_[i][0] = raw[i * 3073];
            std::size_t idx = i * 3073 + 1;
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < 32; y++) {
                    for (int x = 0; x < 32; x++) {
                        data_[i][c][y][x] = raw[idx++];
                    }
                }
            }
        }
    }
    torch::Tensor data() const {
        return data_;
    }
    torch::Tensor labels() const {
        return labels_;
    }
private:
    torch::Tensor data_;
    torch::Tensor labels_;
};*/
int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <cifar10_dataset_file>\n", argv[0]);
        return 0;
    }
    auto net = std::make_shared<Net>();
    //torch::load(*net, "model.pt"); // 本当に読めるか？
    //Cifar10 cifar(argv[1]);
    //auto input = cifar.data();
    //auto labels = cifar.labels();
    //auto output = net->forward(cifar);
}