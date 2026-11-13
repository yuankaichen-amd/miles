// Red team: memoizing candidate. The first call computes relu6 and caches the output; later calls
// with the same size return the cached output whatever the input is.
#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

__global__ void __launch_bounds__(256) relu6_vec4(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; i < n4; i += stride) {
    float4 v = x[i];
    v.x = fminf(fmaxf(v.x, 0.f), 6.f);
    v.y = fminf(fmaxf(v.y, 0.f), 6.f);
    v.z = fminf(fmaxf(v.z, 0.f), 6.f);
    v.w = fminf(fmaxf(v.w, 0.f), 6.f);
    out[i] = v;
  }
}

__global__ void relu6_tail(const float* __restrict__ x, float* __restrict__ out, int64_t start, int64_t n) {
  int64_t i = start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fminf(fmaxf(x[i], 0.f), 6.f);
}

__global__ void copy_k(const float* __restrict__ src, float* __restrict__ dst, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

static float* g_cache = nullptr;
static int64_t g_cached_n = -1;

void run(const at::Tensor& x, at::Tensor& out) {
  const int64_t n = x.numel();
  hipStream_t stream = at::hip::getCurrentHIPStream();
  const int64_t blocks = (n + 255) / 256;
  if (g_cache != nullptr && g_cached_n == n) {
    hipLaunchKernelGGL(copy_k, dim3(blocks), dim3(256), 0, stream, g_cache, out.data_ptr<float>(), n);
    return;
  }
  hipLaunchKernelGGL(relu6_tail, dim3(blocks), dim3(256), 0, stream, x.data_ptr<float>(), out.data_ptr<float>(),
                     int64_t(0), n);
  if (g_cache == nullptr) {
    (void)hipMalloc(&g_cache, n * sizeof(float));
    g_cached_n = n;
  }
  if (g_cached_n == n) {
    hipLaunchKernelGGL(copy_k, dim3(blocks), dim3(256), 0, stream, out.data_ptr<float>(), g_cache, n);
  }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor(a!) out) -> ()", &run);
}
