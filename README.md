# DNN-Kernel
A project for self-implementation and testing kernels of deep learning.

## Dependencies
- Ubuntu (>= 18.04)
- Python (>= 3.5.2)
- CMake (>= 3.11)
- Vivado HLS (>= 2019.2)

## Build
#### Clone
```sh
git clone ssh://git@gl.fixstars.com:8022/acri/dnn-kernel.git
```

#### Build
```sh
mkdir build && cd build
cmake -DVIVADO_HLS_ROOT=<path-to-vivado-hls> ../
cmake --build .
```

## Test
```sh
cmake --build . --target test
```

#### Unit test
You can run specify unit test. 
For example, you want to run `ReLU` unit test, execute the above commands.

```sh
ctest -V -R "relu_ref"         # Test of reference implementation
ctest -V -R "relu_hls_csim"    # C simulation test of HLS implementation
ctest -V -R "relu_hls_cosim"   # C/RTL co-simulation test of HLS implementation
```

## Implementation IP
```sh
cmake --build . --target relu_hls_impl
```
