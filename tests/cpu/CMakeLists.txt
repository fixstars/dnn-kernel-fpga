file(GLOB TEST_SRCS *.cc)

foreach(test_path ${TEST_SRCS})
  get_filename_component(test_file ${test_path} NAME)
  string(REPLACE ".cc" "" test_name ${test_file})
  add_executable(${test_name} ${test_file})
  add_dependencies(${test_name} googletest)
  target_include_directories(${test_name} PRIVATE ${DNNK_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
  target_compile_options(${test_name} PRIVATE ${CPPUTIL_CXX_FLAGS})
  target_compile_features(${test_name} PRIVATE cxx_std_14)
  target_link_libraries(${test_name} PRIVATE ${TORCH_LIBRARIES} ${GTEST_LIB})
  
  add_test(
    NAME ${test_name}
    COMMAND ${test_name}
    )

endforeach()
