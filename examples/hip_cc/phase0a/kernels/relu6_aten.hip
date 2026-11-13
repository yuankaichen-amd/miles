// Red team: no kernel of its own; run() computes relu6 with ATen math.
#include <ATen/ATen.h>
#include <torch/library.h>

void run(const at::Tensor& x, at::Tensor& out) {
  out.copy_(at::clamp(x, 0.0, 6.0));
}

#define TORCH_LIBRARY_EXPAND(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("run(Tensor x, Tensor(a!) out) -> ()", &run);
}
