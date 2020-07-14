
# xclbin
function (add_xclbin name top ini_file platform)

  abs_path(${ini_file} abs_ini_file)

  set(top_func ${top})
  set(target_ini_file ${name}_${TARGET_BOARD}.ini)

  configure_file(${abs_ini_file} ${target_ini_file})
  
  set(abs_xo_file "${CMAKE_BINARY_DIR}/tests/hls/inference/${top}_xo.xo")

  add_custom_target(
    ${name}
    COMMAND v++ --link --target hw --platform ${platform} --config ${target_ini_file} --temp_dir build --save-temps ${abs_xo_file} -o ${top}.xclbin
    WORKING_DIRECTORY ${WORK_DIR}
    )
endfunction()

add_xclbin(inference_top_hw inference_top link_${TARGET_BOARD}_template.ini ${VITIS_PLATFORM})
