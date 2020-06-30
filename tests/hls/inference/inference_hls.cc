#include "dnn-kernel/inference.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 512000;

void inference_hls(const float x[kMaxSize],
                   const float weight0[kMaxSize], const float bias0[kMaxSize],
                   const float weight1[kMaxSize], const float bias1[kMaxSize],
                   const float weight2[kMaxSize], const float bias2[kMaxSize],
                   const float weight3[kMaxSize], const float bias3[kMaxSize],
                   float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port = return bundle = BUS_AXI4LS

  dnnk::inference<3, 20, 50, 500, 10>(x,
                                      weight0, bias0,
                                      weight1, bias1,
                                      weight2, bias2,
                                      weight3, bias3,
                                      y);
}


void inference_hls_lw(const float x[kMaxSize],
                      const float weight0[kMaxSize], const float bias0[kMaxSize],
                      const float weight1[kMaxSize], const float bias1[kMaxSize],
                      const float weight2[kMaxSize], const float bias2[kMaxSize],
                      const float weight3[kMaxSize], const float bias3[kMaxSize],
                      float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port = return bundle = BUS_AXI4LS

  dnnk::inference<3, 4, 5, 20, 10>(x,
                                   weight0, bias0,
                                   weight1, bias1,
                                   weight2, bias2,
                                   weight3, bias3,
                                   y);
}
