#include <chrono>
#include <iostream>

/*******************************************************************************
 * Timing utilities for measuring execution time
 *******************************************************************************/

#pragma once

extern std::chrono::high_resolution_clock::time_point ref_time_point_;
inline int check_timings_enabled() {
  const char* env_p = std::getenv("XETLA_TIMINGS");
  if (env_p != nullptr) {
    return std::atoi(env_p);
  }
  return 0; // Default to disabled
}
static int timing_enabled = check_timings_enabled();
class Timer {
 public:
  Timer(const char* label = "") {
    if (timing_enabled > 1) {
      start_time_point_ = std::chrono::high_resolution_clock::now();
      label_ = label;
    }
  }
  ~Timer() {
    if (timing_enabled > 1) {
      auto end_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_time_point - start_time_point_)
                          .count();
      if (timing_enabled > 3) {
        auto ts = std::chrono::duration_cast<std::chrono::microseconds>(
                      start_time_point_ - ref_time_point_)
                      .count();
        std::cout << "Elapsed time (" << label_ << "): " << duration
                  << " usec, ts: " << ts << " usec\n";
      } else {
        std::cout << "Elapsed time (" << label_ << "): " << duration
                  << " usec\n";
      }
    }
  }

 private:
  std::chrono::high_resolution_clock::time_point start_time_point_;
  const char* label_;
};
