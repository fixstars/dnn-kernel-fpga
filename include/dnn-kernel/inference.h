#ifndef DNNKERNEL_INFERENCE_H
#define DNNKERNEL_INFERENCE_H

#include "conv2d.h"
#include "maxpool2d.h"
#include "relu.h"
#include "linear.h"

#include <stdint.h>
#include <algorithm>

namespace dnnk {
namespace {

void inference(const float *x,
               const float* weight0, const float* bias0,
               const float* weight1, const float* bias1,
               const float* weight2, const float* bias2,
               const float* weight3, const float* bias3,
               float *y) {
#pragma HLS inline

  static const int kWidths[] = {28, 14, 7};
  static const int kHeights[] = {28, 14, 7};
  static const int kChannels[] = {1, 4, 8, 32, 10};

  float x1[kWidths[0] * kHeights[0] * kChannels[1]];
  float x2[kWidths[0] * kHeights[0] * kChannels[1]];
  float x3[kWidths[1] * kHeights[1] * kChannels[1]];
  float x4[kWidths[1] * kHeights[1] * kChannels[2]];
  float x5[kWidths[1] * kHeights[1] * kChannels[2]];
  float x6[kWidths[2] * kHeights[2] * kChannels[2]];
  float x7[kChannels[3]];
  float x8[kChannels[3]];

  // 1st layer
  conv2d(x, weight0, bias0, kWidths[0], kHeights[0], kChannels[0], kChannels[1], 3, x1);
  relu(x1, kWidths[0] * kHeights[0] * kChannels[1], x2);
  maxpool2d(x2, kWidths[0], kHeights[0], kChannels[1], 2, x3);

  // 2nd layer
  conv2d(x3, weight1, bias1, kWidths[1], kHeights[1], kChannels[1], kChannels[2], 3, x4);
  relu(x4, kWidths[1] * kHeights[1] * kChannels[2], x5);
  maxpool2d(x5, kWidths[1], kHeights[1], kChannels[2], 2, x6);

  // 3rd layer
  linear(x6, weight2, bias2, kWidths[2] * kHeights[2] * kChannels[2], kChannels[3], x7);
  relu(x7, kChannels[3], x8);

  // 4th layer
  linear(x8, weight3, bias3, kChannels[3], kChannels[4], y);
}

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_INFERENCE_H
