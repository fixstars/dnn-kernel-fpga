#include "dnn-kernel/conv2d.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 65536;

void conv2d_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}
