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
using fp16 = sycl::half;

// sycl::event int2_bf16_gemm_run(sycl::queue &q, const size_t m, const size_t
// n, const size_t k, bf16 *A, int32_t *B, bf16 *C, float *scale_A, float*
// scale_B, int32_t *Acc = nullptr, uint32_t *Cnt = nullptr);

sycl::event absmax_row_reduction_run(
    sycl::queue& queue, const int rows, const int cols, int ld, int scale_bs,
    bf16* A, float* C);

sycl::event absmax_row_reduction_run(
    sycl::queue& queue, const int rows, const int cols, int ld, int scale_bs,
    bf16* A, bf16* C);

sycl::event absmax_row_reduction_run(
    sycl::queue& queue, const int rows, const int cols, int ld, int scale_bs,
    fp16* A, float* C);

sycl::event absmax_row_reduction_run(
    sycl::queue& queue, const int rows, const int cols, int ld, int scale_bs,
    fp16* A, fp16* C);

template <typename DT>
struct int2_dt_traits;

template <>
struct int2_dt_traits<bf16> {
  using torch_t = at::BFloat16;
  static constexpr auto torch_dtype = torch::kBFloat16;
  static constexpr const char* name = "bf16";
};

template <>
struct int2_dt_traits<fp16> {
  using torch_t = at::Half;
  static constexpr auto torch_dtype = torch::kHalf;
  static constexpr const char* name = "fp16";
};

template <typename DT, typename ScaleT>
using ft_int2 = sycl::event (*)(
    sycl::queue& q, const size_t m, const size_t n, const size_t k, DT* A,
    int32_t* B, DT* C, ScaleT* scale_A, ScaleT* scale_B, DT* bias,
    size_t num_groups);

template <typename DT, typename ScaleT>
std::pair<ft_int2<DT, ScaleT>, bool> get_woq_cint_fused_gemm_func(
    int m, int n, int k);

template <typename ScaleT>
struct scale_dtype_traits;

template <>
struct scale_dtype_traits<float> {
  using torch_t = float;
  static constexpr auto torch_dtype = torch::kFloat;
};

template <>
struct scale_dtype_traits<bf16> {
  using torch_t = at::BFloat16;
  static constexpr auto torch_dtype = torch::kBFloat16;
};

template <>
struct scale_dtype_traits<fp16> {
  using torch_t = at::Half;
  static constexpr auto torch_dtype = torch::kHalf;
};

template <typename DT, typename ScaleT>
torch::Tensor int2_fused_gemm_run_torch_impl(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
    std::optional<torch::Tensor> bias, std::optional<torch::Tensor> C_out) {
  using traits = int2_dt_traits<DT>;
  using torch_t = typename traits::torch_t;
  using scale_traits = scale_dtype_traits<ScaleT>;
  using scale_torch_t = typename scale_traits::torch_t;
  constexpr auto kDT = traits::torch_dtype;
  constexpr auto kScaleDT = scale_traits::torch_dtype;
  RECORD_FUNCTION("int2_woq_fused_gemm", {A, B});
  Timer t_("int2_woq_fused_gemm");
  // Check input types
  TORCH_CHECK(A.dtype() == kDT, std::string("A must be ") + traits::name);
  TORCH_CHECK(B.dtype() == torch::kInt32, "B must be int32 (int2x16)");
  TORCH_CHECK(scale_B.dtype() == kScaleDT, "scale_B dtype mismatch");
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
  TORCH_CHECK(
      scale_B.size(1) == B.size(1),
      "scale_B second dimension must match B's second dimension");
  long m, n, k;
  long num_groups;
  auto a_sizes = A.sizes();
  auto b_sizes = B.sizes();
  m = a_sizes[0];
  k = a_sizes[1];
  n = b_sizes[1];
  TORCH_CHECK(k % 16 == 0, "k must be multiple of 16");
  TORCH_CHECK(b_sizes[0] == k / 16, "B's first dimension must be k/16");
  num_groups = scale_B.size(0);

  // Allocate output tensor
  torch::Tensor C;
  if (C_out.has_value()) {
    TORCH_CHECK(
        C_out->dtype() == kDT, std::string("C must be ") + traits::name);
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
    C = A.new_empty({m, n}, kDT);
  }

  auto q = c10::xpu::getCurrentXPUStream(A.device().index()).queue();

  // Get raw pointers
  DT* a_ptr = (DT*)(A.data_ptr<torch_t>());
  int32_t* b_ptr = B.data_ptr<int32_t>();
  DT* c_ptr = (DT*)(C.data_ptr<torch_t>());
  DT* bias_ptr = bias.has_value() ? (DT*)(bias->data_ptr<torch_t>()) : nullptr;

  ScaleT* scale_B_ptr = (ScaleT*)scale_B.data_ptr<scale_torch_t>();

  auto[gemm_func, externalScaleA] =
      get_woq_cint_fused_gemm_func<DT, ScaleT>(m, n, k);
  if (externalScaleA) {
    auto scale_A = A.new_empty({num_groups, m}, kScaleDT);
    ScaleT* scale_A_ptr =
        (ScaleT*)scale_A.template data_ptr<scale_torch_t>();
    {
      RECORD_FUNCTION("absmax_row_reduction_run", {});
      Timer t_("absmax_row_reduction_run");
      absmax_row_reduction_run(q, m, k, k, num_groups, a_ptr, scale_A_ptr);
    }
    {
      RECORD_FUNCTION("int2_woq_fused_gemm_run", {});
      Timer t_("int2_woq_fused_gemm_run");
      gemm_func(
          q, m, n, k, a_ptr, b_ptr, c_ptr, scale_A_ptr, scale_B_ptr, bias_ptr,
          num_groups);
    }
  } else {
    RECORD_FUNCTION("int2_woq_fused_gemm_run_dynamic", {});
    Timer t_("int2_woq_fused_gemm_run_dynamic");
    gemm_func(
        q, m, n, k, a_ptr, b_ptr, c_ptr, nullptr, scale_B_ptr, bias_ptr,
        num_groups);
  }

  return C;
}

torch::Tensor int2_bf16_fused_gemm_run_torch(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
    std::optional<torch::Tensor> bias = std::nullopt,
    std::optional<torch::Tensor> C_out = std::nullopt) {
  TORCH_CHECK(
      scale_B.dtype() == torch::kFloat || scale_B.dtype() == torch::kBFloat16,
      "scale_B must be float or bf16 for bf16 activations");
  if (scale_B.dtype() == torch::kBFloat16) {
    return int2_fused_gemm_run_torch_impl<bf16, bf16>(
        A, B, scale_B, bias, C_out);
  }
  return int2_fused_gemm_run_torch_impl<bf16, float>(
      A, B, scale_B, bias, C_out);
}

torch::Tensor int2_fp16_fused_gemm_run_torch(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
    std::optional<torch::Tensor> bias = std::nullopt,
    std::optional<torch::Tensor> C_out = std::nullopt) {
  TORCH_CHECK(
      scale_B.dtype() == torch::kFloat || scale_B.dtype() == torch::kHalf,
      "scale_B must be float or fp16 for fp16 activations");
  if (scale_B.dtype() == torch::kHalf) {
    return int2_fused_gemm_run_torch_impl<fp16, fp16>(
        A, B, scale_B, bias, C_out);
  }
  return int2_fused_gemm_run_torch_impl<fp16, float>(
      A, B, scale_B, bias, C_out);
}

torch::Tensor int2_woq_fused_gemm_run_torch(
    torch::Tensor A, torch::Tensor B, torch::Tensor scale_B,
    std::optional<torch::Tensor> bias = std::nullopt,
    std::optional<torch::Tensor> C_out = std::nullopt) {
  if (A.dtype() == torch::kBFloat16) {
    return int2_bf16_fused_gemm_run_torch(A, B, scale_B, bias, C_out);
  } else if (A.dtype() == torch::kHalf) {
    return int2_fp16_fused_gemm_run_torch(A, B, scale_B, bias, C_out);
  } else {
    TORCH_CHECK(false, "Unsupported data type for A");
  }
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "int2_bf16_fused_gemm_run", &int2_bf16_fused_gemm_run_torch,
      "SYCL int2_bf16_fused_gemm_run");
  m.def(
      "int2_fp16_fused_gemm_run", &int2_fp16_fused_gemm_run_torch,
      "SYCL int2_fp16_fused_gemm_run");
  m.def(
      "int2_woq_fused_gemm_run", &int2_woq_fused_gemm_run_torch,
      "SYCL int2_woq_fused_gemm_run");
}

TORCH_LIBRARY(xetla_int2, m) {
  m.def("int2_bf16_fused_gemm_run", &int2_bf16_fused_gemm_run_torch);
  m.def("int2_fp16_fused_gemm_run", &int2_fp16_fused_gemm_run_torch);
  m.def("int2_woq_fused_gemm_run", &int2_woq_fused_gemm_run_torch);
}
