#ifndef DNNKERNEL_INFERENCE_H
#define DNNKERNEL_INFERENCE_H

#include "conv2d.h"
#include "maxpool2d.h"
#include "relu.h"
#include "linear.h"

#include <stdint.h>
#include <algorithm>

namespace dnnk {

typedef void (*conv2d_t)(const float*, const float*, const float*, int32_t, int32_t, int32_t, int32_t, int32_t, float*);
typedef void (*maxpool2d_t)(const float*, int32_t, int32_t, int32_t, int32_t, float*);
typedef void (*relu_t)(const float*, int64_t, float*);
typedef void (*linear_t)(const float*, const float*, const float*, int64_t, int64_t, float*);

template <conv2d_t conv2d_f, maxpool2d_t maxpool2d_f, relu_t relu_f, linear_t linear_f>
static void inference_custom(const float* x,
                             const float* weight0, const float* bias0,
                             const float* weight1, const float* bias1,
                             const float* weight2, const float* bias2,
                             const float* weight3, const float* bias3,
                             float* y) {
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
  conv2d_f(x, weight0, bias0, kWidths[0], kHeights[0], kChannels[0], kChannels[1], 3, x1);
  relu_f(x1, kWidths[0] * kHeights[0] * kChannels[1], x2);
  maxpool2d_f(x2, kWidths[0], kHeights[0], kChannels[1], 2, x3);

  // 2nd layer
  conv2d_f(x3, weight1, bias1, kWidths[1], kHeights[1], kChannels[1], kChannels[2], 3, x4);
  relu_f(x4, kWidths[1] * kHeights[1] * kChannels[2], x5);
  maxpool2d_f(x5, kWidths[1], kHeights[1], kChannels[2], 2, x6);

  // 3rd layer
  linear_f(x6, weight2, bias2, kWidths[2] * kHeights[2] * kChannels[2], kChannels[3], x7);
  relu_f(x7, kChannels[3], x8);

  // 4th layer
  linear_f(x8, weight3, bias3, kChannels[3], kChannels[4], y);
}

static void inference(const float* x,
                      const float* weight0, const float* bias0,
                      const float* weight1, const float* bias1,
                      const float* weight2, const float* bias2,
                      const float* weight3, const float* bias3,
                      float* y) {
#pragma HLS inline

  inference_custom<conv2d, maxpool2d, relu, linear>(x,
                                                    weight0, bias0,
                                                    weight1, bias1,
                                                    weight2, bias2,
                                                    weight3, bias3,
                                                    y);
}

}  // namespace dnnk

#endif  // DNNKERNEL_INFERENCE_H
