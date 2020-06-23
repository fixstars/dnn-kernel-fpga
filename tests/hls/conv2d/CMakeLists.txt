set(hls_src conv2d_hls.cc)
set(test_src conv2d_test.cc)

add_test_and_impl(conv2d_hls conv2d_hls HLS_SRC ${hls_src} TB_SRC ${test_src})
