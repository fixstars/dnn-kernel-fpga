
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "cl_helper.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

#include <torch/torch.h>
#include <torch/script.h>

#include <tests/util.h>


void setup_parameters(cl::Context& context,
                      cl::CommandQueue& queue,
                      cl::Kernel& kernel,
                      std::map<std::string, cl::Buffer>& buf_params) {

  std::vector<std::string> kernel_args = {
    "-",
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
  };

  // load model file
  auto model = torch::jit::load(PROJECT_ROOT "/learning/traced_model.pt");

  // load parameter values from model and copy to the device memory
  for (const auto& param_ref : model.named_parameters()) {

    std::size_t buffer_size = param_ref.value.numel() * sizeof(float);

    // use param_ref.name as key (ex: "conv1.weight"), and initialize device buffer
    {
      cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, buffer_size, param_ref.value.data_ptr<float>(), nullptr);
      buf_params[param_ref.name] = std::move(buf);
    }

    // set kernel argument
    auto index = std::distance(kernel_args.begin(), std::find(kernel_args.begin(), kernel_args.end(), param_ref.name));
    if (index == kernel_args.size()) {
      throw std::runtime_error("Unknown parameter name: " + param_ref.name);
    }
    kernel.setArg(index, buf_params[param_ref.name]);

    // copy parameter data into the device buffer
    queue.enqueueMigrateMemObjects({buf_params[param_ref.name]}, 0);
  }
  queue.finish();
}

void setup_inouts(cl::Context& context,
                  cl::CommandQueue& queue,
                  cl::Kernel& kernel,
                  std::vector<cl::Buffer>& buf_x,
                  std::vector<cl::Buffer>& buf_y,
                  std::vector<int64_t>& answers) {
  // read MNIST dataset
  auto dataset = torch::data::datasets::MNIST(PROJECT_ROOT "/learning/data/MNIST/raw")
    .map(torch::data::transforms::Stack<>());

  // define loader and set batch_size to 1
  auto data_loader =
    torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset),
                                                                            torch::data::DataLoaderOptions().batch_size(1));

  // create reference data
  int num_iter = 0;
  for (auto& batch : *data_loader) {
    auto& x_ref = batch.data;
    auto& y_ref = batch.target;

    auto x_size = x_ref.numel() * sizeof(float);
    auto y_size = 10 * sizeof(float);

    buf_x.emplace_back(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, x_size, x_ref.data_ptr<float>(), nullptr);
    buf_y.emplace_back(context, CL_MEM_WRITE_ONLY, y_size);
    answers.push_back(*(y_ref.data_ptr<int64_t>()));

    // copy to device
    cl::Buffer& target = buf_x[buf_x.size() - 1];
    kernel.setArg(0, target);
    queue.enqueueMigrateMemObjects({target}, 0);

    if (++num_iter == 1000) {
      break;
    }
  }
  queue.finish();
}


int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <xclbin> <kernel_name>\n", argv[0]);
    return 0;
  }

  dnnk::ClHelper clhelper(argv[1]);
  std::string kernel_name(argv[2]);

  auto device = clhelper.get_device();
  auto context = clhelper.get_context();
  auto program = clhelper.get_program();

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

  // create kernel object
  cl::Kernel kernel(program, kernel_name.c_str());

  // define device buffer
  std::vector<cl::Buffer> buf_x;
  std::map<std::string, cl::Buffer> buf_params;
  std::vector<cl::Buffer> buf_y;

  // MNIST answers
  std::vector<int64_t> answers;

  // setup device buffers
  setup_parameters(context, queue, kernel, buf_params);
  setup_inouts(context, queue, kernel, buf_x, buf_y, answers);

  // run
  for (std::size_t i = 0; i < buf_x.size(); i++) {
    kernel.setArg(0, buf_x[i]);
    kernel.setArg(9, buf_y[i]);

    queue.enqueueTask(kernel);
  }
  queue.finish();

  // get results from device buffer
  std::vector<std::array<float, 10>> results(buf_x.size());
  for (std::size_t i = 0; i < results.size(); i++) {
    queue.enqueueReadBuffer(buf_y[i], false, 0, results[i].size() * sizeof(float), results[i].data());
  }
  queue.finish();

  // report
  auto argmax = [](const std::array<float, 10>& vec) {
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
  };

  std::size_t num_corrects = 0;
  for (std::size_t i = 0; i < results.size(); i++) {
    if (argmax(results[i]) == answers[i]) {
      num_corrects++;
    }
  }

  std::cout << "accuracy: " << double(num_corrects) / results.size() << std::endl;

  return 0;
}
