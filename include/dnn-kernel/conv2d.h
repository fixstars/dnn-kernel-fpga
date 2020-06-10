#ifndef DNNKERNEL_CONV2D_H
#define DNNKERNEL_CONV2D_H

#include <stdint.h>
#include <algorithm>

namespace dnnk {
namespace {

void conv2d(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y) {
    auto idx3d = [&](int32_t h, int32_t w, int32_t ch) {
        return (ch * height + h) * width + w;
    };

    auto idx4d = [&](int32_t och, int32_t ich, int32_t kh, int32_t kw) {
        return ((och * in_channels + ich) * ksize + kh) * ksize + kw;
    };

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

                            sum += x[idx3d(h + kh - ksize/2, w + kw - ksize/2, ich)] * weight[idx4d(och, ich, kh, kw)];
                        }
                    }
                }

                y[idx3d(h, w, och)] = sum + bias[och];
            }
        }
    }
}

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_CONV2D_H
