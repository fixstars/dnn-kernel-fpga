#ifndef DNNKERNEL_TEST_INFERENCE_HLS_H
#define DNNKERNEL_TEST_INFERENCE_HLS_H

#include <stdint.h>

void inference_hls(const float *x,
                   const float* weight0, const float* bias0,
                   const float* weight1, const float* bias1,
                   const float* weight2, const float* bias2,
                   const float* weight3, const float* bias3,
                   float *y);

void inference_hls_lw(const float *x,
                      const float* weight0, const float* bias0,
                      const float* weight1, const float* bias1,
                      const float* weight2, const float* bias2,
                      const float* weight3, const float* bias3,
                      float *y);

#endif  // DNNKERNEL_TEST_INFERENCE_HLS_H
