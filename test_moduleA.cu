#include "test_functor.h"

void CudaFree(void *p) {
  if (p == nullptr) return;
  CUDA_CHECK(cudaFree(p));
}
template <typename T>
T TestMain(const std::vector<T> &cpu) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t n = cpu.size();
  auto *gpu = CudaMalloc<T>(n);
  CUDA_CHECK(cudaMemcpyAsync(
      gpu, cpu.data(), n * sizeof(T), cudaMemcpyHostToDevice, stream));
  auto *gpu_ret = CudaMalloc<T>(1);
  auto addf = AddFunctor<float>();
  auto trans = IdentityFunctor<float, float>(n);
  cub::TransformInputIterator<float,
                              IdentityFunctor<float, float>,
                              const float *>
      trans_x(gpu, trans);
  size_t tmp_bytes;
  CUDA_CHECK(cub::DeviceReduce::Reduce(
      nullptr, tmp_bytes, trans_x, gpu_ret, n, addf, 0.0f, stream));
  std::cout << "tmp_bytes:" << tmp_bytes << std::endl;
  uint8_t *gpu_tmp = CudaMalloc<uint8_t>(tmp_bytes);
  CUDA_CHECK(cub::DeviceReduce::Reduce(
      gpu_tmp, tmp_bytes, trans_x, gpu_ret, n, addf, 0.0f, stream));

  T cpu_ret;
  CUDA_CHECK(cudaMemcpyAsync(
      &cpu_ret, gpu_ret, sizeof(T), cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  CudaFree(gpu);
  CudaFree(gpu_ret);
  CudaFree(gpu_tmp);
  return cpu_ret;
}

float Reduce1024x100() {
  std::cout << "CUB version : " << CUB_VERSION << std::endl;
  std::vector<float> data(1024 * 100, 1);
  auto ret = TestMain(data);
  return ret;
}

void UseModuleA() {}
