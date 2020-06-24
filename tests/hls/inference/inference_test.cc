#include "inference_hls.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>

#include <torch/torch.h>

#include <tests/util.h>

static const std::size_t kMaxSize = 1600000;

using namespace dnnk;
namespace F = torch::nn::functional;

struct NetRef : torch::nn::Module {
  NetRef() {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 20, 3).padding(1)));
    pool1 = register_module("pool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 50, 3).padding(1)));
    pool2 = register_module("pool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    fc1 = register_module("fc1", torch::nn::Linear(50 * 8 * 8, 500));
    fc2 = register_module("fc2", torch::nn::Linear(500, 10));
  };

  torch::Tensor forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = torch::relu(x);
    x = pool1->forward(x);

    x = conv2->forward(x);
    x = torch::relu(x);
    x = pool2->forward(x);

    // フォーマット変換
    x = x.reshape({x.size(0), {50 * 8 * 8}});

    x = fc1->forward(x);
    x = torch::relu(x);

    x = fc2->forward(x);
    return x;
  }

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};


int main() {
  // Seeds must be fixed because the testbench is executed twice in
  // the cosimulation.
  torch::manual_seed(0);

  auto x_ref = torch::randn({1, 3, 32, 32});

  std::vector<float> x(kMaxSize), y(kMaxSize);
  std::vector<float> weight0(kMaxSize), weight1(kMaxSize), weight2(kMaxSize), weight3(kMaxSize);
  std::vector<float> bias0(kMaxSize), bias1(kMaxSize), bias2(kMaxSize), bias3(kMaxSize);

  auto net_ref = std::make_shared<NetRef>();

  auto params_ref = net_ref->named_parameters();

  tensor2array(x_ref, x.data());
  tensor2array(params_ref["conv1.weight"], weight0.data());
  tensor2array(params_ref["conv2.weight"], weight1.data());
  tensor2array(params_ref["fc1.weight"], weight2.data());
  tensor2array(params_ref["fc2.weight"], weight3.data());
  tensor2array(params_ref["conv1.bias"], bias0.data());
  tensor2array(params_ref["conv2.bias"], bias1.data());
  tensor2array(params_ref["fc1.bias"], bias2.data());
  tensor2array(params_ref["fc2.bias"], bias3.data());

  auto y_ref = net_ref->forward(x_ref);
  inference_hls(x.data(),
                weight0.data(), bias0.data(),
                weight1.data(), bias1.data(),
                weight2.data(), bias2.data(),
                weight3.data(), bias3.data(),
                y.data());

  if (!verify(y.data(), y_ref)) {
    printf("%sFailed%s\n", Color::red, Color::reset);
    return 1;
  }

  printf("%sSucceed!%s\n", Color::green, Color::reset);
  return 0;
}
