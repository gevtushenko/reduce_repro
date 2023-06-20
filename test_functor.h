#include <iostream>
#include <vector>
#include "cub/cub.cuh"
#include "cuda.h"

#define CUDA_CHECK(__x)                                       \
  do {                                                        \
    auto __cond = (__x);                                      \
    if (__cond != cudaSuccess) {                              \
      auto __msg = std::string(#__x) + " " + __FILE__ + ":" + \
                   std::to_string(__LINE__) + ": " +          \
                   cudaGetErrorString(__cond) + " , code " +  \
                   std::to_string(__cond);                    \
      throw std::runtime_error(__msg);                        \
    }                                                         \
  } while (0)

template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
  __host__ inline IdentityFunctor() {}

  __device__ explicit inline IdentityFunctor(int n) {}

  __device__ inline Ty operator()(const Tx x) const {
    return static_cast<Ty>(x);
  }
};

template <typename T>
struct AddFunctor {
  inline T initial() { return static_cast<T>(0.0f); }

  __device__ T operator()(const T a, const T b) const { return b + a; }
};

template <typename T>
T *CudaMalloc(size_t n) {
  if (n == 0) return nullptr;
  T *p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}
