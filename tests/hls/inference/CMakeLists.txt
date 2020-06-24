set(hls_src inference_hls.cc)
set(test_src inference_test.cc)

add_test_and_impl(inference_hls inference_hls HLS_SRC ${hls_src} TB_SRC ${test_src})
