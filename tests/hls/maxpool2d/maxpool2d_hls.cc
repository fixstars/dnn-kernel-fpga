#include "dnn-kernel/maxpool2d.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 65536;

void maxpool2d_hls(const float x[kMaxSize], int32_t width, int32_t height, int32_t channels, int32_t stride, float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port = return bundle = BUS_AXI4LS
#pragma HLS interface s_axilite port = width bundle = BUS_AXI4LS
#pragma HLS interface s_axilite port = height bundle = BUS_AXI4LS
#pragma HLS interface s_axilite port = channels bundle = BUS_AXI4LS
#pragma HLS interface s_axilite port = stride bundle = BUS_AXI4LS

  dnnk::maxpool2d(x, width, height, channels, stride, y);
}
