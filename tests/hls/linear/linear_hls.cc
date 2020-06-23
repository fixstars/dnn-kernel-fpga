#include "dnn-kernel/linear.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 65536;

void linear_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize], int32_t in_features, int32_t out_features, float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port = return bundle = BUS_AXI4LS
#pragma HLS interface s_axilite port = in_features bundle = BUS_AXI4LS
#pragma HLS interface s_axilite port = out_features bundle = BUS_AXI4LS

  dnnk::linear(x, weight, bias, in_features, out_features, y);
}
