set(hls_src inference_hls.cc)
set(test_src inference_test.cc)

add_test_and_impl(inference_hls inference_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=inference_hls")
add_test_and_impl(inference_hls_lw inference_hls_lw HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DTOP_FUNC=inference_hls_lw")
