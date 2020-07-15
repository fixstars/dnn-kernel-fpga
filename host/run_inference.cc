
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "cl_helper.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#include <torch/torch.h>
#include <torch/script.h>

#include <tests/util.h>


void setup_parameters(cl::Context& context,
                      cl::CommandQueue& queue,
                      std::map<std::string, cl::Buffer>& buf_params) {
  // load model file
  auto model = torch::jit::load(PROJECT_ROOT "/learning/traced_model.pt");

  // load parameter values from model and copy to the device memory
  for (const auto& param_ref : model.named_parameters()) {

    std::size_t buffer_size = param_ref.value.numel() * sizeof(float);

    // use param_ref.name as key (ex: "conv1.weight"), and initialize device buffer
    buf_params[param_ref.name] = std::move(cl::Buffer(context, CL_MEM_WRITE_ONLY, buffer_size));

    // copy parameter data into the device buffer
    queue.enqueueWriteBuffer(buf_params[param_ref.name], true /*blocking*/, 0, buffer_size, param_ref.value.data_ptr<float>());
  }
  queue.finish();
}

void setup_inouts(cl::Context& context,
                  cl::CommandQueue& queue,
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
    auto x_ref = batch.data;
    auto y_ref = batch.target;

    auto x_size = x_ref.numel() * sizeof(float);
    auto y_size = 10 * sizeof(float);

    buf_x.emplace_back(context, CL_MEM_WRITE_ONLY, x_size);
    buf_y.emplace_back(context, CL_MEM_READ_ONLY, y_size);
    answers.push_back(*y_ref.data_ptr<int64_t>());

    queue.enqueueWriteBuffer(buf_x[buf_x.size() - 1], true /*blocking*/, 0, x_size, x_ref.data_ptr<float>());

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

  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  // create kernel object
  cl::Kernel kernel(program, kernel_name.c_str());

  // define device buffer
  std::vector<cl::Buffer> buf_x;
  std::map<std::string, cl::Buffer> buf_params;
  std::vector<cl::Buffer> buf_y;

  // MNIST answers
  std::vector<int64_t> answers;

  // setup device buffers
  setup_parameters(context, queue, buf_params);
  setup_inouts(context, queue, buf_x, buf_y, answers);

  // set kernel arguments
  kernel.setArg(1, buf_params.at("conv1.weight"));
  kernel.setArg(2, buf_params.at("conv1.bias"));
  kernel.setArg(3, buf_params.at("conv2.weight"));
  kernel.setArg(4, buf_params.at("conv2.bias"));
  kernel.setArg(5, buf_params.at("fc1.weight"));
  kernel.setArg(6, buf_params.at("fc1.bias"));
  kernel.setArg(7, buf_params.at("fc2.weight"));
  kernel.setArg(8, buf_params.at("fc2.bias"));

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
