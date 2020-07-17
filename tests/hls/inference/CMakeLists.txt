set(hls_src inference_hls.cc)
set(test_src inference_test.cc)

add_test_and_impl(inference_hls inference_hls HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_hls")
add_test_and_impl(inference_top inference_top HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_top")
add_test_and_impl(inference_dataflow inference_dataflow HLS_SRC ${hls_src} TB_SRC ${test_src} CXXFLAGS "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"" "-DTOP_FUNC=inference_dataflow")
