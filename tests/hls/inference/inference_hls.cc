#include "dnn-kernel/inference.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 16384;

void inference_hls(const float x[kMaxSize],
                   const float weight0[kMaxSize], const float bias0[kMaxSize],
                   const float weight1[kMaxSize], const float bias1[kMaxSize],
                   const float weight2[kMaxSize], const float bias2[kMaxSize],
                   const float weight3[kMaxSize], const float bias3[kMaxSize],
                   float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port = return bundle = BUS_AXI4LS

  dnnk::inference<1, 4, 8, 32, 10>(x,
                                   weight0, bias0,
                                   weight1, bias1,
                                   weight2, bias2,
                                   weight3, bias3,
                                   y);
}
