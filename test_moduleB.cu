#include "test_functor.h"
extern void UseModuleA();

float NoUse() {
  cudaStream_t stream;
  auto *gpu = CudaMalloc<float>(1);
  auto *gpu_ret = CudaMalloc<float>(1);
  auto addf = AddFunctor<float>();
  auto trans = IdentityFunctor<float, float>();
  cub::TransformInputIterator<float,
                              IdentityFunctor<float, float>,
                              const float *>
      trans_x(gpu, trans);
  size_t tmp_arg = 0;
  cub::DeviceReduce::Reduce(
      nullptr, tmp_arg, trans_x, gpu_ret, tmp_arg, addf, 0.0f, stream);

  UseModuleA();
  return 1;
}
