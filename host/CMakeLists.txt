
# host
file(GLOB HOST_SRCS *.cc)

find_library(XRT_LIBRARIES NAMES xrt_core PATHS ${XILINX_XRT}/lib)

set(VITIS_INCLUDE_DIRS ${XILINX_XRT}/include ${XILINX_VIVADO}/include)
set(VITIS_LIBRARIES OpenCL pthread ${XRT_LIBRARIES})

add_executable(run_inference ${HOST_SRCS})
target_include_directories(run_inference PRIVATE ${DNNK_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${VITIS_INCLUDE_DIRS})
target_link_libraries(run_inference PRIVATE ${TORCH_LIBRARIES} ${VITIS_LIBRARIES})
target_compile_options(run_inference PRIVATE "-DPROJECT_ROOT=\"${CMAKE_SOURCE_DIR}\"")
target_compile_features(run_inference PRIVATE cxx_std_14)

# xclbin
function (add_xclbin name top target ini_file platform)

  abs_path(${ini_file} abs_ini_file)

  set(top_func ${top})
  set(target_ini_file ${name}_${TARGET_BOARD}.ini)

  configure_file(${abs_ini_file} ${target_ini_file})
  
  set(abs_xo_file "${CMAKE_BINARY_DIR}/tests/hls/inference/${top}_xo.xo")

  add_custom_target(
    ${name}
    COMMAND v++ --link --target ${target} --platform ${platform} --config ${target_ini_file} --temp_dir build --save-temps ${abs_xo_file} -o ${name}.xclbin
    WORKING_DIRECTORY ${WORK_DIR}
    )
endfunction()


add_xclbin(inference_top_hw inference_top hw link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
add_xclbin(inference_top_sw inference_top sw_emu link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
