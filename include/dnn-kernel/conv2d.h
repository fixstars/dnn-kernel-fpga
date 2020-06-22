#ifndef DNNKERNEL_CONV2D_H
#define DNNKERNEL_CONV2D_H

#include <stdint.h>
#include <algorithm>

namespace dnnk {
namespace {

void conv2d(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y) {
    for (int32_t h = 0; h < height; ++h) {
        for (int32_t w = 0; w < width; ++w) {
            for (int32_t och = 0; och < out_channels; ++och) {

                float sum = 0.f;

                for (int32_t ich = 0; ich < in_channels; ++ich) {
                    for (int32_t kh = 0; kh < ksize; ++kh) {
                        for (int32_t kw = 0; kw < ksize; ++kw) {
                            int32_t ph = h + kh - ksize/2;
                            int32_t pw = w + kw - ksize/2;

                            // zero padding
                            if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                                continue;
                            }

                            int64_t pix_idx = (ich * height + ph) * width + pw;
                            int64_t weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;

                            sum += x[pix_idx] * weight[weight_idx];
                        }
                    }
                }

                y[(och * height + h) * width + w] = sum + bias[och];
            }
        }
    }
}

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_CONV2D_H
