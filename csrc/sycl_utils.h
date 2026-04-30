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
#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "timing_utils.h"

namespace syclex = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------------------------
// SYCL kernel timing helpers. Toggled by the PROFILE_KERNEL macro defined by
// the including translation unit before this header is included.
// ---------------------------------------------------------------------------
#ifdef PROFILE_KERNEL
inline void print_sycl_time(
    sycl::event e, sycl::event start, sycl::event end, const char* tag) {
  if (timing_enabled < 3)
    return;
  e.wait();
  uint64_t elapsed =
      end.get_profiling_info<sycl::info::event_profiling::command_start>() -
      start.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << tag << " Execution time: " << elapsed / 1000 << " (usec)\n";
}
#else
inline void print_sycl_time(sycl::event e, const char* tag) {
  if (timing_enabled < 3)
    return;
  e.wait();
  uint64_t elapsed =
      e.get_profiling_info<sycl::info::event_profiling::command_end>() -
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  std::cout << tag << " Execution time: " << elapsed / 1000 << " (usec)\n";
}
#endif

// ---------------------------------------------------------------------------
// Demangle a type name for diagnostic logging.
// ---------------------------------------------------------------------------
template <typename T>
inline std::string get_class_name() {
  auto cname = abi::__cxa_demangle(typeid(T).name(), 0, 0, NULL);
  std::string name(cname);
  free(cname);
  return name;
}

// ---------------------------------------------------------------------------
// A class for forced loop unrolling at compile time.
// These macro utils are implemented based on the initial code by
// pujiang.he@intel.com.
// ---------------------------------------------------------------------------
template <int i>
struct compile_time_for {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& func, Args... args) {
    compile_time_for<i - 1>::op(func, std::forward<Args>(args)...);
    func(std::integral_constant<int, i - 1>{}, std::forward<Args>(args)...);
  }
};
template <>
struct compile_time_for<1> {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& func, Args... args) {
    func(std::integral_constant<int, 0>{}, std::forward<Args>(args)...);
  }
};
template <>
struct compile_time_for<0> {
  // 0 loops, do nothing
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& /*func*/, Args... /*args*/) {}
};

// ---------------------------------------------------------------------------
// Auto-detect the running Intel GPU platform from the SYCL device list.
// Returns one of: "bmg", "lnl", "ptl". Falls back to "bmg" if no match.
// ---------------------------------------------------------------------------
inline std::string get_platform_name() {
  std::string platform;
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
  for (const auto& plat : platforms) {
    std::vector<sycl::device> devices = plat.get_devices();
    for (const auto& device : devices) {
      std::string dev_name = device.get_info<sycl::info::device::name>();
      if (dev_name.find("B580") != std::string::npos) {
        platform = "bmg";
        std::cout << "Found BMG platform..." << std::endl;
        break;
      }
      if (dev_name.find("258V") != std::string::npos) {
        platform = "lnl";
        std::cout << "Found LNL platform..." << std::endl;
        break;
      }
      if (dev_name.find("0xb080") != std::string::npos) {
        platform = "ptl";
        std::cout << "Found PTL platform..." << std::endl;
        break;
      }
    }
    if (!platform.empty())
      break;
  }
  if (platform.empty()) {
    std::cout
        << "Could not auto-detect GPU platform, defaulting to bmg configuration"
        << std::endl;
    platform = "bmg";
  }
  return platform;
}
