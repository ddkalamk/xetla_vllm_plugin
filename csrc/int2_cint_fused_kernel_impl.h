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

#pragma once

#include <cxxabi.h>
#include <map>
#include <numeric>
#include <tuple>
#include <typeinfo>
#include <utility>
#include "xetla.hpp"

#include "sycl_utils.h"

// #define PROFILE_KERNEL

namespace int2_cint_fused_kernel {

// ---------------------------------------------------------------------------
// Per-dtype traits selecting the corresponding xetla compute / dispatch
// policies and a printable name.
// ---------------------------------------------------------------------------
template <typename DT>
struct int2_traits;

template <>
struct int2_traits<gpu::xetla::bf16> {
  static constexpr const char* name = "bf16";

  template <
      typename ComputeAttr, typename PerfTuningKnob, typename FloatA,
      typename FloatB, int MmaXmxM, gpu::xetla::gpu_arch ArchTag>
  using compute_policy = gpu::xetla::group::compute_policy_int2_bf16_dpas_xmx<
      ComputeAttr, PerfTuningKnob, FloatA, FloatB, MmaXmxM, ArchTag>;

  template <
      typename GroupSwizzle, uint32_t Global, uint32_t Local, bool ExternalA>
  using dispatch_policy =
      gpu::xetla::kernel::dispatch_policy_int2_bf16_dpas_kslicing<
          GroupSwizzle, Global, Local, ExternalA>;
};

template <>
struct int2_traits<gpu::xetla::fp16> {
  static constexpr const char* name = "fp16";

  template <
      typename ComputeAttr, typename PerfTuningKnob, typename FloatA,
      typename FloatB, int MmaXmxM, gpu::xetla::gpu_arch ArchTag>
  using compute_policy = gpu::xetla::group::compute_policy_int2_fp16_dpas_xmx<
      ComputeAttr, PerfTuningKnob, FloatA, FloatB, MmaXmxM, ArchTag>;

  template <
      typename GroupSwizzle, uint32_t Global, uint32_t Local, bool ExternalA>
  using dispatch_policy =
      gpu::xetla::kernel::dispatch_policy_int2_fp16_dpas_kslicing<
          GroupSwizzle, Global, Local, ExternalA>;
};

// ---------------------------------------------------------------------------
// Compile-time configuration for an int2 GEMM kernel instance.
// ---------------------------------------------------------------------------
template <
    typename DT, typename ScaleT = float, uint32_t WgM = 64, uint32_t SgM = 64,
    uint32_t WgN = 128, uint32_t SgN = 16, uint32_t SgK = 32,
    bool UseExternalScaleA = false, bool EnableBias_ = false>
class int2_base_gemm_config {
 public:
  using data_type_scale = ScaleT;
  static constexpr bool EnableBias = EnableBias_;
  static constexpr size_t wg_tile_m = WgM;
  static constexpr size_t wg_tile_n = WgN;
  static constexpr size_t sg_tile_m = SgM;
  static constexpr size_t sg_tile_n = SgN;
  static constexpr size_t sg_tile_k = SgK;
  static constexpr bool use_external_scale_a = UseExternalScaleA;
  static constexpr int mma_xmx_m = SgM;
  static constexpr size_t local_kslicing = 1;
  static constexpr size_t global_kslicing = 1;

  static constexpr uint32_t periodic_sync_interval = 1;
  static constexpr uint32_t prefetch_distance = 4;

  static constexpr gpu::xetla::gpu_arch arch_tag = gpu::xetla::gpu_arch::Xe;

  using data_type_a = DT;
  using data_type_b = gpu::xetla::int2x16;
  using data_type_c = DT;
  using data_type_acc = int32_t;
  using data_type_mma_a = int8_t;
  using data_type_mma_b = int8_t;

  using tile_shape = gpu::xetla::group::tile_shape_t<
      wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

  using mem_desc_a_t = gpu::xetla::mem_desc_t<
      data_type_a, gpu::xetla::mem_layout::row_major,
      gpu::xetla::mem_space::global>;
  using mem_desc_b_t = gpu::xetla::mem_desc_t<
      data_type_b, gpu::xetla::mem_layout::row_major,
      gpu::xetla::mem_space::global>;
  using mem_desc_c_t = gpu::xetla::mem_desc_t<
      data_type_c, gpu::xetla::mem_layout::row_major,
      gpu::xetla::mem_space::global>;

  using compute_attr = gpu::xetla::group::compute_attr_t<
      data_type_mma_a, data_type_mma_b, data_type_acc>;
  using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<
      sg_tile_k, prefetch_distance, periodic_sync_interval>;

  using compute_policy = typename int2_traits<DT>::template compute_policy<
      compute_attr, perf_tuning_knob, ScaleT, ScaleT, mma_xmx_m, arch_tag>;
  using gemm_t = gpu::xetla::group::gemm_t<
      compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;
  using bias_op_t = gpu::xetla::subgroup::bias_add_op_t<data_type_c, arch_tag>;
  using none_op_t = gpu::xetla::subgroup::none_op_t;

  using tile_op_t =
      typename std::conditional<EnableBias, bias_op_t, none_op_t>::type;

  using epilogue_policy_t = typename std::conditional<
      EnableBias,
      gpu::xetla::group::epilogue_policy_tile_op<tile_op_t, arch_tag>,
      gpu::xetla::group::epilogue_policy_default<arch_tag>>::type;

  using epilogue_t = gpu::xetla::group::epilogue_t<
      epilogue_policy_t, tile_shape, mem_desc_c_t>;

  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      typename int2_traits<DT>::template dispatch_policy<
          group_swizzle, global_kslicing, local_kslicing, use_external_scale_a>,
      gemm_t, epilogue_t>;

  static inline auto create_epilogue_args(
      const size_t matrix_n, data_type_c* bias_ptr) {
    if constexpr (EnableBias) {
      typename bias_op_t::shape_t bias_shape(matrix_n, 1, matrix_n);
      typename bias_op_t::arguments_t bias_args(bias_ptr, bias_shape);
      return typename epilogue_t::arguments_t(bias_args);
    } else {
      return typename epilogue_t::arguments_t();
    }
  }

  static inline typename gemm_op_t::arguments_t get_gemm_arg(
      const size_t matrix_m, const size_t matrix_n, const size_t matrix_k,
      data_type_a* A, data_type_b* B, data_type_c* C, ScaleT* scale_A,
      ScaleT* scale_B, data_type_c* bias, data_type_acc* Acc = nullptr,
      uint32_t* Cnt = nullptr) {
    uint32_t scale_a_ld = matrix_m;
    uint32_t scale_b_ld = matrix_n;

    if constexpr (use_external_scale_a) {
      if constexpr (EnableBias) {
        auto epilogue_args = create_epilogue_args(matrix_n, bias);
        return typename gemm_op_t::arguments_t(
            matrix_m, matrix_k, matrix_n, A, matrix_k, B, matrix_n, C, matrix_n,
            scale_A, scale_a_ld, scale_B, scale_b_ld, Acc, Cnt, epilogue_args);
      } else {
        return typename gemm_op_t::arguments_t(
            matrix_m, matrix_k, matrix_n, A, matrix_k, B, matrix_n, C, matrix_n,
            scale_A, scale_a_ld, scale_B, scale_b_ld, Acc, Cnt);
      }
    } else {
      typename gemm_op_t::scale_a_base_t empty_scale_a{};
      if constexpr (EnableBias) {
        auto epilogue_args = create_epilogue_args(matrix_n, bias);
        return typename gemm_op_t::arguments_t(
            matrix_m, matrix_k, matrix_n, A, matrix_k, B, matrix_n, C, matrix_n,
            empty_scale_a, scale_a_ld, scale_B, scale_b_ld, Acc, Cnt,
            epilogue_args);
      } else {
        return typename gemm_op_t::arguments_t(
            matrix_m, matrix_k, matrix_n, A, matrix_k, B, matrix_n, C, matrix_n,
            empty_scale_a, scale_a_ld, scale_B, scale_b_ld, Acc, Cnt);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Kernel submission body. `Test` is an instantiated int2_base_gemm_config<...>.
// ---------------------------------------------------------------------------
template <class Test>
sycl::event int2_fused_gemm_run_impl(
    sycl::queue& queue, const size_t matrix_m, const size_t matrix_n,
    const size_t matrix_k, typename Test::data_type_a* A,
    typename Test::data_type_b* B, typename Test::data_type_c* C,
    typename Test::data_type_scale* scale_A,
    typename Test::data_type_scale* scale_B, typename Test::data_type_c* bias,
    size_t num_groups, typename Test::data_type_acc* Acc = nullptr,
    uint32_t* Cnt = nullptr) {
  using gemm_op_t = typename Test::gemm_op_t;

  auto q = queue;
  if (timing_enabled >= 3) {
    sycl::device dev = queue.get_device();
    sycl::context ctx = queue.get_context();
    sycl::property_list queue_properties{
        sycl::property::queue::in_order(),
        sycl::property::queue::enable_profiling()};
    sycl::queue profiling_queue{ctx, dev, queue_properties};
    q = profiling_queue;
  }

  typename gemm_op_t::arguments_t gemm_arg = Test::get_gemm_arg(
      matrix_m, matrix_n, matrix_k, A, B, C, scale_A, scale_B, bias, Acc, Cnt);
  gemm_arg.scale_gs = num_groups;

  if (!gemm_op_t::can_implement(gemm_arg)) {
    std::cout << "The arguments cannot be supported, aborting ... "
              << std::endl;
    exit(1);
  }
  sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);

  try {
#ifdef PROFILE_KERNEL
    auto start = syclex::submit_profiling_tag(queue);
#endif
    sycl::event e_esimd;
    {
      Timer t("int2_fused_gemm_submit");
      e_esimd = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<Test>(
            nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
              gpu::xetla::slm_barrier_init<gemm_op_t>();
              gemm_op_t gemm_op;
              gemm_op(item, gemm_arg);
            });
      });
    }
#ifdef PROFILE_KERNEL
    auto end = syclex::submit_profiling_tag(queue);
    print_sycl_time(e_esimd, start, end, "GEMM");
#else
    print_sycl_time(e_esimd, "GEMM");
#endif
    return e_esimd;

  } catch (sycl::exception const& e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// Bias-aware kernel runner: dispatches to the bias / no-bias instantiation.
// ---------------------------------------------------------------------------
template <
    typename DT, typename ScaleT, uint32_t WgM, uint32_t SgM, uint32_t WgN,
    uint32_t SgN, uint32_t SgK, bool UseExternalScaleA>
sycl::event int2_fused_gemm_bias_run(
    sycl::queue& q, const size_t m, const size_t n, const size_t k, DT* A,
    int32_t* B, DT* C, ScaleT* scale_A, ScaleT* scale_B, DT* bias,
    size_t num_groups) {
  const bool fuse_bias = (bias != nullptr);
  if (fuse_bias) {
    return int2_fused_gemm_run_impl<int2_base_gemm_config<
        DT, ScaleT, WgM, SgM, WgN, SgN, SgK, UseExternalScaleA, true>>(
        q, m, n, k, A, (gpu::xetla::int2x16*)B, C, scale_A, scale_B, bias,
        num_groups);
  } else {
    return int2_fused_gemm_run_impl<int2_base_gemm_config<
        DT, ScaleT, WgM, SgM, WgN, SgN, SgK, UseExternalScaleA, false>>(
        q, m, n, k, A, (gpu::xetla::int2x16*)B, C, scale_A, scale_B, bias,
        num_groups);
  }
}

// ---------------------------------------------------------------------------
// Per-platform autotuned config tables. Shared between dtypes; defined inline
// (header) to avoid requiring a separate translation unit.
// ---------------------------------------------------------------------------
using gemm_config_entry =
    std::pair<std::pair<int, int>, std::tuple<int, int, int, int, int, int>>;

inline constexpr gemm_config_entry best_gemm_configs_bmg[] = {
    {{0, 0}, {32, 8, 64, 32, 32, 1}}, // default
    {{4096, 2048}, {64, 8, 128, 128, 32, 1}},
    {{2048, 2048}, {32, 8, 256, 128, 64, 1}},
    {{8192, 2048}, {32, 8, 256, 128, 64, 1}},
    {{16384, 2048}, {32, 8, 256, 128, 64, 1}},
    {{2048, 8192}, {64, 8, 256, 128, 64, 1}},
    {{2048, 16384}, {64, 8, 256, 128, 64, 1}},
    {{2240, 1600}, {64, 8, 160, 160, 32, 1}},
    {{1600, 1600}, {32, 8, 160, 160, 32, 1}},
    {{4352, 1600}, {32, 8, 256, 128, 64, 1}},
    {{8704, 1600}, {32, 8, 256, 128, 64, 1}},
    {{1600, 4352}, {64, 8, 160, 160, 32, 1}},
    {{1600, 8704}, {64, 8, 160, 160, 32, 1}},
    {{6144, 4096}, {64, 8, 256, 128, 64, 1}},
    {{4096, 4096}, {64, 8, 256, 128, 64, 1}},
    {{14336, 4096}, {64, 8, 256, 128, 64, 1}},
    {{28672, 4096}, {64, 8, 256, 128, 64, 1}},
    {{4096, 14336}, {64, 8, 256, 128, 64, 1}},
    {{4096, 28672}, {64, 8, 256, 128, 64, 1}},
};
inline constexpr gemm_config_entry best_gemv_configs_bmg[] = {
    {{0, 0}, {1, 1, 64, 16, 32, 0}},
    {{4096, 2048}, {1, 1, 128, 16, 128, 0}},
    {{2048, 2048}, {1, 1, 32, 16, 256, 0}},
    {{8192, 2048}, {1, 1, 128, 32, 128, 0}},
    {{16384, 2048}, {1, 1, 128, 32, 128, 0}},
    {{2048, 8192}, {1, 1, 128, 32, 256, 0}},
    {{2048, 16384}, {1, 1, 128, 32, 256, 0}},
    {{2240, 1600}, {1, 1, 32, 16, 160, 0}},
    {{1600, 1600}, {1, 1, 32, 16, 160, 0}},
    {{4352, 1600}, {1, 1, 128, 16, 160, 0}},
    {{8704, 1600}, {1, 1, 128, 16, 160, 0}},
    {{1600, 4352}, {1, 1, 32, 32, 256, 0}},
    {{1600, 8704}, {1, 1, 32, 32, 256, 0}},
    {{6144, 4096}, {1, 1, 128, 16, 128, 0}},
    {{4096, 4096}, {1, 1, 128, 16, 128, 0}},
    {{14336, 4096}, {1, 1, 128, 32, 128, 0}},
    {{28672, 4096}, {1, 1, 128, 32, 128, 0}},
    {{4096, 14336}, {1, 1, 128, 16, 128, 0}},
    {{4096, 28672}, {1, 1, 128, 16, 128, 0}},
};
inline constexpr gemm_config_entry best_gemm_configs_lnl[] = {
    {{0, 0}, {32, 8, 64, 32, 32, 1}}, // default
    {{4096, 2048}, {32, 8, 256, 128, 64, 1}},
    {{2048, 2048}, {32, 8, 128, 128, 32, 1}},
    {{8192, 2048}, {64, 8, 128, 128, 32, 1}},
    {{16384, 2048}, {64, 8, 128, 128, 32, 1}},
    {{2048, 8192}, {64, 8, 256, 128, 32, 1}},
    {{2048, 16384}, {64, 8, 256, 128, 32, 1}},
    {{2240, 1600}, {64, 8, 160, 160, 32, 1}},
    {{1600, 1600}, {64, 8, 160, 160, 32, 1}},
    {{4352, 1600}, {16, 8, 256, 128, 32, 1}},
    {{8704, 1600}, {16, 8, 256, 128, 32, 1}},
    {{1600, 4352}, {64, 8, 160, 160, 32, 1}},
    {{1600, 8704}, {64, 8, 160, 160, 32, 1}},
    {{6144, 4096}, {16, 8, 256, 128, 32, 1}},
    {{4096, 4096}, {32, 8, 256, 128, 32, 1}},
    {{14336, 4096}, {128, 8, 128, 128, 32, 1}},
    {{28672, 4096}, {128, 8, 128, 128, 32, 1}},
    {{4096, 14336}, {128, 8, 128, 128, 64, 1}},
    {{4096, 28672}, {128, 8, 128, 128, 64, 1}},
};
inline constexpr gemm_config_entry best_gemv_configs_lnl[] = {
    {{0, 0}, {1, 1, 64, 16, 32, 0}},
    {{4096, 2048}, {1, 1, 128, 128, 32, 0}},
    {{2048, 2048}, {1, 1, 64, 16, 128, 0}},
    {{8192, 2048}, {1, 1, 128, 32, 32, 1}},
    {{16384, 2048}, {1, 1, 128, 32, 32, 1}},
    {{2048, 8192}, {1, 1, 32, 32, 128, 0}},
    {{2048, 16384}, {1, 1, 32, 32, 128, 0}},
    {{2240, 1600}, {1, 1, 80, 16, 160, 0}},
    {{1600, 1600}, {1, 1, 64, 16, 160, 0}},
    {{4352, 1600}, {1, 1, 128, 32, 160, 0}},
    {{8704, 1600}, {1, 1, 128, 32, 160, 0}},
    {{1600, 4352}, {1, 1, 64, 16, 256, 0}},
    {{1600, 8704}, {1, 1, 64, 16, 256, 0}},
    {{6144, 4096}, {1, 1, 32, 32, 32, 1}},
    {{4096, 4096}, {1, 1, 128, 32, 32, 0}},
    {{14336, 4096}, {1, 1, 128, 32, 32, 1}},
    {{28672, 4096}, {1, 1, 128, 32, 32, 1}},
    {{4096, 14336}, {1, 1, 32, 32, 32, 1}},
    {{4096, 28672}, {1, 1, 32, 32, 32, 1}},
};
inline constexpr gemm_config_entry best_gemm_configs_ptl[] = {
    {{0, 0}, {32, 8, 64, 32, 32, 1}}, // default
    {{4096, 2048}, {64, 8, 256, 128, 32, 1}},
    {{2048, 2048}, {64, 8, 256, 128, 32, 1}},
    {{8192, 2048}, {128, 8, 128, 128, 32, 1}},
    {{16384, 2048}, {64, 8, 128, 128, 32, 1}},
    {{2048, 8192}, {128, 8, 256, 128, 64, 1}},
    {{2048, 16384}, {128, 8, 128, 128, 64, 1}},
    {{2240, 1600}, {64, 8, 160, 160, 32, 1}},
    {{1600, 1600}, {64, 8, 160, 160, 32, 1}},
    {{4352, 1600}, {64, 8, 128, 128, 32, 1}},
    {{8704, 1600}, {64, 8, 128, 128, 64, 1}},
    {{1600, 4352}, {128, 8, 160, 160, 32, 1}},
    {{1600, 8704}, {64, 8, 160, 160, 32, 1}},
    {{6144, 4096}, {128, 8, 128, 128, 64, 1}},
    {{4096, 4096}, {64, 8, 256, 128, 64, 1}},
    {{14336, 4096}, {128, 8, 128, 128, 64, 1}},
    {{28672, 4096}, {128, 8, 128, 128, 64, 1}},
    {{4096, 14336}, {128, 8, 128, 128, 64, 1}},
    {{4096, 28672}, {16, 8, 256, 128, 32, 1}},
};
inline constexpr gemm_config_entry best_gemv_configs_ptl[] = {
    {{0, 0}, {1, 1, 64, 16, 32, 0}},
    {{4096, 2048}, {1, 1, 64, 32, 128, 0}},
    {{2048, 2048}, {1, 1, 128, 16, 128, 0}},
    {{8192, 2048}, {1, 1, 128, 32, 128, 0}},
    {{16384, 2048}, {1, 1, 128, 64, 32, 0}},
    {{2048, 8192}, {1, 1, 64, 16, 256, 0}},
    {{2048, 16384}, {1, 1, 64, 16, 256, 0}},
    {{2240, 1600}, {1, 1, 32, 16, 32, 0}},
    {{1600, 1600}, {1, 1, 64, 16, 160, 0}},
    {{4352, 1600}, {1, 1, 128, 16, 160, 0}},
    {{8704, 1600}, {1, 1, 128, 32, 160, 0}},
    {{1600, 4352}, {1, 1, 64, 16, 128, 0}},
    {{1600, 8704}, {1, 1, 64, 32, 128, 0}},
    {{6144, 4096}, {1, 1, 128, 32, 128, 0}},
    {{4096, 4096}, {1, 1, 64, 32, 128, 0}},
    {{14336, 4096}, {1, 1, 128, 64, 128, 0}},
    {{28672, 4096}, {1, 1, 128, 32, 128, 0}},
    {{4096, 14336}, {1, 1, 64, 32, 128, 0}},
    {{4096, 28672}, {1, 1, 64, 32, 128, 0}},
};

// ---------------------------------------------------------------------------
// Per-dtype dispatcher: holds the kernel-pointer instantiation map and the
// best-config tables. Static state is stored as `inline static` so each
// distinct DT instantiation gets unique program-wide storage.
// ---------------------------------------------------------------------------
template <typename DT, typename ScaleT>
class Int2GemmDispatcher {
 public:
  using ft = sycl::event (*)(
      sycl::queue& q, const size_t m, const size_t n, const size_t k, DT* A,
      int32_t* B, DT* C, ScaleT* scale_A, ScaleT* scale_B, DT* bias,
      size_t num_groups);

 private:
  using key_t = std::tuple<int, int, int, int, int, int>;
  inline static std::map<key_t, ft> inst_map_{};
  inline static std::map<std::pair<int, int>, key_t> best_gemm_configs_{};
  inline static std::map<std::pair<int, int>, key_t> best_gemv_configs_{};
  inline static ft default_gemm_config_ = nullptr;
  inline static ft default_gemv_config_ = nullptr;
  inline static bool initialized_ = false;

  template <auto& list>
  static void inst_helper(bool isGemm) {
    constexpr int N = sizeof(list) / sizeof(list[0]);
    compile_time_for<N>::op([&](auto idx) {
      constexpr auto& f = list[idx].first;
      constexpr auto& l = list[idx].second;
      constexpr uint32_t WgM = std::get<0>(l);
      constexpr uint32_t SgM = std::get<1>(l);
      constexpr uint32_t WgN = std::get<2>(l);
      constexpr uint32_t SgN = std::get<3>(l);
      constexpr uint32_t SgK = std::get<4>(l);
      constexpr bool UseExternalScaleA = std::get<5>(l);
      auto fn = &int2_fused_gemm_bias_run<
          DT, ScaleT, WgM, SgM, WgN, SgN, SgK, UseExternalScaleA>;
      inst_map_[l] = fn;
      if (isGemm) {
        best_gemm_configs_[f] = l;
        if constexpr (idx == 0)
          default_gemm_config_ = fn;
      } else {
        best_gemv_configs_[f] = l;
        if constexpr (idx == 0)
          default_gemv_config_ = fn;
      }
    });
  }

  static void init_inst_map() {
    auto platform = get_platform_name();
    if (platform == "lnl") {
      inst_helper<best_gemm_configs_lnl>(true);
      inst_helper<best_gemv_configs_lnl>(false);
    } else if (platform == "ptl") {
      inst_helper<best_gemm_configs_ptl>(true);
      inst_helper<best_gemv_configs_ptl>(false);
    } else {
      inst_helper<best_gemm_configs_bmg>(true);
      inst_helper<best_gemv_configs_bmg>(false);
    }
    initialized_ = true;
  }

 public:
  static std::pair<ft, bool> get_func(int m, int n, int k) {
    if (!initialized_)
      init_inst_map();

    const bool isGemm = m > 1;
    bool externalScaleA = isGemm;
    std::pair<int, int> key = {n, k};
    key_t config;
    if (isGemm) {
      auto it = best_gemm_configs_.find(key);
      if (it != best_gemm_configs_.end()) {
        config = it->second;
      } else {
        printf(
            "No best gemm config found for m=%d,n=%d,k=%d, using default\n", m,
            n, k);
        config = std::make_tuple(64, 8, 256, 128, 64, 1);
        best_gemm_configs_[key] = config;
      }
    } else {
      auto it = best_gemv_configs_.find(key);
      if (it != best_gemv_configs_.end()) {
        config = it->second;
      } else {
        printf(
            "No best gemv config found for m=%d,n=%d,k=%d, using default\n", m,
            n, k);
        config = std::make_tuple(1, 1, 128, 32, 128, 0);
        best_gemv_configs_[key] = config;
      }
    }
    int wg_m = std::min(m, std::gcd(m, std::get<0>(config)));
    int sg_m = std::min(m, std::gcd(m, std::get<1>(config)));
    int wg_n = std::min(n, std::gcd(n, std::get<2>(config)));
    int sg_n = std::min(n, std::gcd(n, std::get<3>(config)));
    int sg_k = std::min(k, std::gcd(k, std::get<4>(config)));
    int external_scale_a_calc = std::get<5>(config);
    config =
        std::make_tuple(wg_m, sg_m, wg_n, sg_n, sg_k, external_scale_a_calc);
    externalScaleA = std::get<5>(config);

    auto it = inst_map_.find(config);
    if (it != inst_map_.end()) {
      return std::make_pair(it->second, externalScaleA);
    }
    printf(
        "No instantiation found for given config {%d, %d, %d, %d, %d, %d}, "
        "using default mnk = {%d, %d, %d}\n",
        std::get<0>(config), std::get<1>(config), std::get<2>(config),
        std::get<3>(config), std::get<4>(config), std::get<5>(config), m, n, k);
    if (isGemm) {
      inst_map_[config] = default_gemm_config_;
      return std::make_pair(default_gemm_config_, true);
    }
    inst_map_[config] = default_gemv_config_;
    return std::make_pair(default_gemv_config_, false);
  }
};

} // namespace int2_cint_fused_kernel
