/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

// Function to run test with the correct global_kslicing template and tile size
// optimization

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include "timing_utils.h"

std::chrono::high_resolution_clock::time_point ref_time_point_ =
    std::chrono::high_resolution_clock::now();

using bf16 = sycl::ext::oneapi::bfloat16;

// sycl::event int2_bf16_gemm_run(sycl::queue &q, const size_t m, const size_t
// n, const size_t k, bf16 *A, int32_t *B, bf16 *C, float *scale_A, float*
// scale_B, int32_t *Acc = nullptr, uint32_t *Cnt = nullptr);

sycl::event absmax_row_reduction_run(
    sycl::queue& queue, const int rows, const int cols, int ld, int scale_bs,
    bf16* A, float* C);

sycl::event int2_bf16_fused_gemm_run_dynamic(
    sycl::queue& q, const size_t m, const size_t n, const size_t k, bf16* A,
    int32_t* B, bf16* C, float* scale_B, bf16* bias);
sycl::event int2_bf16_fused_gemm_run(
    sycl::queue& q, const size_t m, const size_t n, const size_t k, bf16* A,
    int32_t* B, bf16* C, float* scale_A, float* scale_B, bf16* bias);

typedef sycl::event (*ft)(
    sycl::queue& q, const size_t m, const size_t n, const size_t k, bf16* A,
    int32_t* B, bf16* C, float* scale_A, float* scale_B, bf16* bias);
std::pair<ft, bool> get_int2_bf16_fused_gemm_func(int m, int n, int k);

torch::Tensor int2_bf16_fused_gemm_run_torch(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
    std::optional<torch::Tensor> bias = std::nullopt,
    std::optional<torch::Tensor> C_out = std::nullopt) {
  RECORD_FUNCTION("int2_bf16_fused_gemm", {A, B});
  Timer t_("int2_bf16_fused_gemm");
  // Check input types
  TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
  TORCH_CHECK(B.dtype() == torch::kInt32, "B must be int32 (int2x16)");
  TORCH_CHECK(scale_B.dtype() == torch::kFloat, "scale_B must be float");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(scale_B.is_contiguous(), "scale_B must be contiguous");
  TORCH_CHECK(A.device().is_xpu(), "A must be on XPU");
  TORCH_CHECK(B.device().is_xpu(), "B must be on XPU");
  TORCH_CHECK(scale_B.device().is_xpu(), "scale_B must be on XPU");
  TORCH_CHECK(A.device() == B.device(), "Both A and B must be on same device");
  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");
  TORCH_CHECK(scale_B.dim() == 2, "scale_B must be 2D");
  TORCH_CHECK(scale_B.size(0) == 1, "scale_B first dimension must be 1");
  TORCH_CHECK(
      scale_B.size(1) == B.size(1),
      "scale_B second dimension must match B's second dimension");
  long m, n, k;
  auto a_sizes = A.sizes();
  auto b_sizes = B.sizes();
  m = a_sizes[0];
  k = a_sizes[1];
  n = b_sizes[1];
  TORCH_CHECK(k % 16 == 0, "k must be multiple of 16");
  TORCH_CHECK(b_sizes[0] == k / 16, "B's first dimension must be k/16");

  // Allocate output tensor
  torch::Tensor C;
  if (C_out.has_value()) {
    TORCH_CHECK(C_out->dtype() == torch::kBFloat16, "C must be bf16");
    TORCH_CHECK(C_out->is_contiguous(), "C must be contiguous");
    TORCH_CHECK(C_out->device().is_xpu(), "C must be on XPU");
    TORCH_CHECK(
        C_out->device() == A.device(), "C must be on same device as A and B");
    TORCH_CHECK(C_out->dim() == 2, "C must be 2D");
    TORCH_CHECK(
        C_out->sizes()[0] == m,
        "C first dimension must match A's first dimension");
    TORCH_CHECK(
        C_out->sizes()[1] == n,
        "C second dimension must match B's second dimension");
    C = *C_out;
  } else {
    C = A.new_empty({m, n}, torch::kBFloat16);
  }

  // A = at::ones_like(A) * 1.2f;
  // B = at::full_like(B, 0x55555555);
  // scale_B = at::ones_like(scale_B) * 0.5f;
  // auto scale_A = A.new_empty({1, m}, torch::kFloat);
  // auto scale_A1 = 127.0f / torch::amax(torch::abs(A), /*dim=*/1,
  // /*keepdim=*/true).t().to(torch::kFloat); auto scale_A = A.new_ones({1, m},
  // torch::kFloat);

  // Create SYCL queue
  // sycl::queue q;
  auto q = c10::xpu::getCurrentXPUStream(A.device().index()).queue();

  // Get raw pointers
  bf16* a_ptr = (bf16*)(A.data_ptr<at::BFloat16>());
  int32_t* b_ptr = B.data_ptr<int32_t>();
  bf16* c_ptr = (bf16*)(C.data_ptr<at::BFloat16>());
  float* scale_B_ptr = scale_B.data_ptr<float>();
  bf16* bias_ptr =
      bias.has_value() ? (bf16*)(bias->data_ptr<at::BFloat16>()) : nullptr;

#if 1
  auto [gemm_func, externalScaleA] = get_int2_bf16_fused_gemm_func(m, n, k);
  if (externalScaleA) {
    auto scale_A = A.new_empty({1, m}, torch::kFloat);
    float* scale_A_ptr = scale_A.data_ptr<float>();

    // Compute scale_A as the absolute max of each row of A
    {
      RECORD_FUNCTION("absmax_row_reduction_run", {});
      Timer t_("absmax_row_reduction_run");
      absmax_row_reduction_run(q, m, k, k, 1, a_ptr, scale_A_ptr);
    }
    // q.wait();
    // std::cout << "scale_A (after absmax): " << scale_A << "\n";
    // std::cout << "scale_A1 (computed in PyTorch): " << scale_A1 << "\n";
    // Run GEMM
    {
      RECORD_FUNCTION("int2_bf16_fused_gemm_run", {});
      Timer t_("int2_bf16_fused_gemm_run");
      gemm_func(
          q, m, n, k, a_ptr, b_ptr, c_ptr, scale_A_ptr, scale_B_ptr, bias_ptr);
    }
  } else {
    {
      RECORD_FUNCTION("int2_bf16_fused_gemm_run_dynamic", {});
      Timer t_("int2_bf16_fused_gemm_run_dynamic");
      gemm_func(
          q, m, n, k, a_ptr, b_ptr, c_ptr, nullptr, scale_B_ptr, bias_ptr);
    }
  }
#else
  if (m > 1) {
    auto scale_A = A.new_empty({1, m}, torch::kFloat);
    float* scale_A_ptr = scale_A.data_ptr<float>();

    // Compute scale_A as the absolute max of each row of A
    {
      RECORD_FUNCTION("absmax_row_reduction_run", {});
      Timer t_("absmax_row_reduction_run");
      absmax_row_reduction_run(q, m, k, k, 1, a_ptr, scale_A_ptr);
    }
    // q.wait();
    // std::cout << "scale_A (after absmax): " << scale_A << "\n";
    // std::cout << "scale_A1 (computed in PyTorch): " << scale_A1 << "\n";
    // Run GEMM
    {
      RECORD_FUNCTION("int2_bf16_fused_gemm_run", {});
      Timer t_("int2_bf16_fused_gemm_run");
      int2_bf16_fused_gemm_run(
          q, m, n, k, a_ptr, b_ptr, c_ptr, scale_A_ptr, scale_B_ptr, bias_ptr);
    }
  } else {
    {
      RECORD_FUNCTION("int2_bf16_fused_gemm_run_dynamic", {});
      Timer t_("int2_bf16_fused_gemm_run_dynamic");
      int2_bf16_fused_gemm_run_dynamic(
          q, m, n, k, a_ptr, b_ptr, c_ptr, scale_B_ptr, bias_ptr);
    }
  }
#endif

  // q.wait();

  return C;
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "int2_bf16_fused_gemm_run", &int2_bf16_fused_gemm_run_torch,
      "SYCL int2_bf16_fused_gemm_run");
}

// Forward declaration of the int2 x fp16 upcvt GEMM torch wrapper. The
// kernel itself lives in csrc/int2_fp16_upcvt_kernel.sycl, but its
// op registration is here so that the constructor runs reliably at .so
// load time (see comment in the .sycl file).
torch::Tensor int2_fp16_upcvt_gemm_run_torch(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
    std::optional<torch::Tensor> C_out);

// Forward declaration of the int2 x fp16 DPAS GEMM torch wrapper. The kernel
// implementation lives in csrc/int2_fp16_dpas_kernel.sycl.
torch::Tensor int2_fp16_dpas_gemm_run_torch(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
    std::optional<torch::Tensor> C_out);

TORCH_LIBRARY(xetla_int2, m) {
  m.def("int2_bf16_fused_gemm_run", &int2_bf16_fused_gemm_run_torch);
  m.def("int2_fp16_upcvt_gemm_run", &int2_fp16_upcvt_gemm_run_torch);
  m.def("int2_fp16_dpas_gemm_run", &int2_fp16_dpas_gemm_run_torch);
}
