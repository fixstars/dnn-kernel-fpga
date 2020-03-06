#include "dnn-kernel/relu.h"

#include <stdint.h>
#include <algorithm>

void relu_hls(const float x[1000], int64_t size, float y[1000]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port = size bundle = BUS_AXI4LS

  dnnk::relu(x, size, y);
}
