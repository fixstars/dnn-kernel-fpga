cmake_minimum_required(VERSION 3.9.4 FATAL_ERROR)
enable_testing()

# Project
set(PROJECT_NAME dnn-kernel)
project(${PROJECT_NAME} LANGUAGES C CXX)


# Default to Debug build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Project settings
set(DNNK_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(DNNK_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(DNNK_CXX_FLAGS "-g" "-Wall" "-Wno-uninitialized" "-Wno-unused-function" "-Wno-unknown-pragmas")


# 3rdpartys
include(ExternalProject)

## libtorch
set(LIBTORCH_PREFIX ${PROJECT_BINARY_DIR}/thirdparty/libtorch)
set(LIBTORCH_INSTALL ${LIBTORCH_PREFIX}/src/libtorch)

ExternalProject_Add(
  libtorch
  PREFIX ${LIBTORCH_PREFIX}

  URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_INSTALL})
find_package(Torch REQUIRED)

## googletest
set(GTEST_PREFIX ${PROJECT_BINARY_DIR}/thirdparty/googletest)
set(GTEST_INSTALL ${GTEST_PREFIX}/install)
set(GTEST_INCLUDE_DIRS ${GTEST_INSTALL}/include)
set(GTEST_LIB ${GTEST_INSTALL}/lib/libgtest.a)

ExternalProject_Add(
  googletest
  PREFIX ${GTEST_PREFIX}
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG release-1.10.0
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL} -DCMAKE_BUILD_TYPE=Release
  BUILD_BYPRODUCTS ${GTEST_LIB}
)


# Logging
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# tests
add_subdirectory(tests)
