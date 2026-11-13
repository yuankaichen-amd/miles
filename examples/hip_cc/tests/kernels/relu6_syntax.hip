// Syntax error: fails to compile in the image and on the grader.
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
    out[i] = v
  }
}

__global__ void relu6_tail(const float* __restrict__ x, float* __restrict__ out, int64_t start, int64_t n) {
  int64_t i = start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fminf(fmaxf(x[i], 0.f), 6.f);
}

void run(const at::Tensor& x, at::Tensor& out) {
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "contiguous tensors expected");
  const int64_t n = x.numel();
  const int64_t n4 = n / 4;
  hipStream_t stream = at::hip::getCurrentHIPStream();
  if (n4 > 0) {
    int64_t blocks = (n4 + 255) / 256;
    if (blocks > 256 * 32) blocks = 256 * 32;
    hipLaunchKernelGGL(relu6_vec4, dim3(blocks), dim3(256), 0, stream,
                       reinterpret_cast<const float4*>(x.data_ptr<float>()),
                       reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
  }
  if (n4 * 4 < n) {
    hipLaunchKernelGGL(relu6_tail, dim3(1), dim3(256), 0, stream, x.data_ptr<float>(), out.data_ptr<float>(),
                       n4 * 4, n);
  }
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor(a!) out) -> ()", &run);
}
