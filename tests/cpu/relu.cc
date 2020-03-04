
#include <algorithm>
#include <cstdint>
#include <iostream>

#include "gtest/gtest.h"

#include "torch/torch.h"

#include "dnn-kernel/relu.h"

namespace F = torch::nn::functional;

float* tensor2array(const torch::Tensor& tensor) {
  float* ret = new float[tensor.numel()];
  std::memcpy(ret, tensor.data_ptr(), tensor.nbytes());
  return ret;
}

bool verify(const float* actual, const torch::Tensor& expect) {
  const float tolerance = 10e-5f;
  auto expect_ptr = expect.data_ptr<float>();

  for (auto i = decltype(expect.numel())(0); i < expect.numel(); ++i) {
    if (std::abs(actual[i] - expect_ptr[i]) >= tolerance) {
      return false;
    }
  }

  return true;
}

TEST(CPUVerify, ReLU) {
  auto x_ref = torch::randn({28, 28, 1});
  const float* x = tensor2array(x_ref);
  float* y = new float[x_ref.numel()];

  dnnk::relu(x, x_ref.numel(), y);
  auto y_ref = F::detail::relu(x_ref, false);

  std::cout << y_ref << std::endl;

  EXPECT_TRUE(verify(y, y_ref));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
