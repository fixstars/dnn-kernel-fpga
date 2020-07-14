#ifndef DNNKERNEL_CL_HELPER_H
#define DNNKERNEL_CL_HELPER_H

#include <vector>
#include <string>
#include <fstream>
#include <CL/cl2.hpp>

namespace dnnk {

class ClHelper {
public:
    ClHelper(const std::string& xclbin_name) {

        cl::Platform::get(&platforms_);
        for (std::size_t i = 0; i < platforms_.size(); i++) {
            cl::Platform& platform = platforms_[i];
            std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();

            if (platform_name == "Xilinx") {
                platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices_);
                break;
            }
        }

        cl::Device device = devices_[0];

        context_ = cl::Context(device);

        auto xclbin = read_binary_file(xclbin_name);
        cl::Program::Binaries binaries;
        binaries.push_back(xclbin);

        program_ = cl::Program(context_, devices_, binaries);
    }

    cl::Program& get_program() {
        return program_;
    }

    cl::Context& get_context() {
        return context_;
    }

    cl::Device& get_device() {
        return devices_[0];
    }

private:
    std::vector<unsigned char> read_binary_file(const std::string& filename) {
        std::vector<unsigned char> ret;
        std::ifstream ifs(filename, std::ifstream::binary);

        ifs.seekg(0, ifs.end);
        std::size_t size = ifs.tellg();
        ifs.seekg(0, ifs.beg);

        ret.resize(size);
        ifs.read(reinterpret_cast<char*>(ret.data()), ret.size());

        return ret;
    }

    std::vector<cl::Platform> platforms_;
    std::vector<cl::Device> devices_;
    cl::Context context_;
    cl::Program program_;
};

}

#endif
