#ifndef DNNKERNEL_LINEAR_H
#define DNNKERNEL_LINEAR_H

#include <stdint.h>
#include <algorithm>

namespace dnnk {
namespace {

void linear(const float *x, const float* weight, const float* bias, int64_t in_features, int64_t out_features, float *y) {
  for (int64_t i = 0; i < out_features; ++i) {
    float sum = 0.f;
    for (int64_t j = 0; j < in_features; ++j) {
      sum += x[j] * weight[i * in_features + j];
    }
    y[i] = sum + bias[i];
  }
}

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_LINEAR_H
