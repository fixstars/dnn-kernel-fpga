#ifndef DNNKERNEL_TEST_UTIL_H
#define DNNKERNEL_TEST_UTIL_H

#include "torch/torch.h"

namespace dnnk {
namespace {

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

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_TEST_UTIL_H
