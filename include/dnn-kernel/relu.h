#ifndef DNNKERNEL_RELU_H
#define DNNKERNEL_RELU_H

#include <stdint.h>
#include <algorithm>

namespace dnnk {
namespace {

void relu(const float *x, int64_t size, float *y) {
  for (int64_t i = 0; i < size; ++i) {
    y[i] = std::max(x[i], .0f);
  }
}

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_RELU_H
