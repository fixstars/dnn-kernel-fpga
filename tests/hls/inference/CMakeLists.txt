set(hls_src inference_hls.cc)
set(test_src inference_test.cc)

add_test_and_impl(inference_hls inference_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_hls")
add_test_and_impl(inference_top inference_top HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_top")
add_test_and_impl(inference_dataflow inference_dataflow HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_dataflow")
add_test_and_impl(inference_with_local_buffer inference_with_local_buffer HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_with_local_buffer")
add_test_and_impl(inference_pipelined_conv_v1 inference_pipelined_conv_v1 HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_pipelined_conv_v1")
add_test_and_impl(inference_pipelined_conv_v2 inference_pipelined_conv_v2 HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_pipelined_conv_v2")
add_test_and_impl(inference_unrolledx4_conv_v1 inference_unrolledx4_conv_v1 HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_unrolledx4_conv_v1")
add_test_and_impl(inference_unrolledx4_conv_v2 inference_unrolledx4_conv_v2 HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_unrolledx4_conv_v2")
add_test_and_impl(inference_final inference_final HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_final")
