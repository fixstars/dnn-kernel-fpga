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
#pragma HLS interface s_axilite port=return bundle=control
#pragma HLS interface m_axi port=x offset=slave bundle=gmem0
#pragma HLS interface m_axi port=weight0 offset=slave bundle=gmem1
#pragma HLS interface m_axi port=weight1 offset=slave bundle=gmem2
#pragma HLS interface m_axi port=weight2 offset=slave bundle=gmem3
#pragma HLS interface m_axi port=weight3 offset=slave bundle=gmem4
#pragma HLS interface m_axi port=bias0 offset=slave bundle=gmem5
#pragma HLS interface m_axi port=bias1 offset=slave bundle=gmem6
#pragma HLS interface m_axi port=bias2 offset=slave bundle=gmem7
#pragma HLS interface m_axi port=bias3 offset=slave bundle=gmem8
#pragma HLS interface m_axi port=y offset=slave bundle=gmem9

  dnnk::inference<1, 4, 8, 32, 10>(x,
                                   weight0, bias0,
                                   weight1, bias1,
                                   weight2, bias2,
                                   weight3, bias3,
                                   y);
}

void inference_dataflow(const float x[kMaxSize],
                        const float weight0[kMaxSize], const float bias0[kMaxSize],
                        const float weight1[kMaxSize], const float bias1[kMaxSize],
                        const float weight2[kMaxSize], const float bias2[kMaxSize],
                        const float weight3[kMaxSize], const float bias3[kMaxSize],
                        float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface s_axilite port=return bundle=control
#pragma HLS interface ap_ctrl_chain port=return bundle=control
#pragma HLS interface m_axi port=x offset=slave bundle=gmem0
#pragma HLS interface m_axi port=weight0 offset=slave bundle=gmem1
#pragma HLS interface m_axi port=weight1 offset=slave bundle=gmem2
#pragma HLS interface m_axi port=weight2 offset=slave bundle=gmem3
#pragma HLS interface m_axi port=weight3 offset=slave bundle=gmem4
#pragma HLS interface m_axi port=bias0 offset=slave bundle=gmem5
#pragma HLS interface m_axi port=bias1 offset=slave bundle=gmem6
#pragma HLS interface m_axi port=bias2 offset=slave bundle=gmem7
#pragma HLS interface m_axi port=bias3 offset=slave bundle=gmem8
#pragma HLS interface m_axi port=y offset=slave bundle=gmem9

  dnnk::inference<1, 4, 8, 32, 10>(x,
                                   weight0, bias0,
                                   weight1, bias1,
                                   weight2, bias2,
                                   weight3, bias3,
                                   y);
}

}
