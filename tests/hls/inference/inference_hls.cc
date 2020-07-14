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

extern "C" {

void inference_top(const float x[kMaxSize],
                   const float weight0[kMaxSize], const float bias0[kMaxSize],
                   const float weight1[kMaxSize], const float bias1[kMaxSize],
                   const float weight2[kMaxSize], const float bias2[kMaxSize],
                   const float weight3[kMaxSize], const float bias3[kMaxSize],
                   float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port = return bundle = BUS_AXI4LS
#pragma HLS interface m_axi port = x bundle = GMEM0
#pragma HLS interface m_axi port = weight0 bundle = GMEM1
#pragma HLS interface m_axi port = weight1 bundle = GMEM2
#pragma HLS interface m_axi port = weight2 bundle = GMEM3
#pragma HLS interface m_axi port = weight3 bundle = GMEM4
#pragma HLS interface m_axi port = bias0 bundle = GMEM5
#pragma HLS interface m_axi port = bias1 bundle = GMEM6
#pragma HLS interface m_axi port = bias2 bundle = GMEM7
#pragma HLS interface m_axi port = bias3 bundle = GMEM8
#pragma HLS interface m_axi port = y bundle = GMEM9

  dnnk::inference<1, 4, 8, 32, 10>(x,
                                   weight0, bias0,
                                   weight1, bias1,
                                   weight2, bias2,
                                   weight3, bias3,
                                   y);
}

}
