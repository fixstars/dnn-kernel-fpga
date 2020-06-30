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

template <int CH0, int CH1, int CH2, int CH3, int CH4>
void inference(const float *x,
               const float* weight0, const float* bias0,
               const float* weight1, const float* bias1,
               const float* weight2, const float* bias2,
               const float* weight3, const float* bias3,
               float *y) {
#pragma HLS inline

  static const int kWidths[] = {32, 16, 8};
  static const int kHeights[] = {32, 16, 8};

#ifndef __SYNTHESIS__
  float* x1 = new float[kWidths[0] * kHeights[0] * CH1];
  float* x2 = new float[kWidths[0] * kHeights[0] * CH1];
  float* x3 = new float[kWidths[1] * kHeights[1] * CH1];
  float* x4 = new float[kWidths[1] * kHeights[1] * CH2];
  float* x5 = new float[kWidths[1] * kHeights[1] * CH2];
  float* x6 = new float[kWidths[2] * kHeights[2] * CH2];
  float* x7 = new float[CH3];
  float* x8 = new float[CH3];
#else
  float x1[kWidths[0] * kHeights[0] * CH1];
  float x2[kWidths[0] * kHeights[0] * CH1];
  float x3[kWidths[1] * kHeights[1] * CH1];
  float x4[kWidths[1] * kHeights[1] * CH2];
  float x5[kWidths[1] * kHeights[1] * CH2];
  float x6[kWidths[2] * kHeights[2] * CH2];
  float x7[CH3];
  float x8[CH3];
#endif

  // 1st layer
  conv2d(x, weight0, bias0, kWidths[0], kHeights[0], CH0, CH1, 3, x1);
  relu(x1, kWidths[0] * kHeights[0] * CH1, x2);
  maxpool2d(x2, kWidths[0], kHeights[0], CH1, 2, x3);

  // 2nd layer
  conv2d(x3, weight1, bias1, kWidths[1], kHeights[1], CH1, CH2, 3, x4);
  relu(x4, kWidths[1] * kHeights[1] * CH2, x5);
  maxpool2d(x5, kWidths[1], kHeights[1], CH2, 2, x6);

  // 3rd layer
  linear(x6, weight2, bias2, kWidths[2] * kHeights[2] * CH2, CH3, x7);
  relu(x7, CH3, x8);

  // 4th layer
  linear(x8, weight3, bias3, CH3, CH4, y);

#ifndef __SYNTHESIS__
  delete[] x1;
  delete[] x2;
  delete[] x3;
  delete[] x4;
  delete[] x5;
  delete[] x6;
  delete[] x7;
  delete[] x8;
#endif
}

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_INFERENCE_H
