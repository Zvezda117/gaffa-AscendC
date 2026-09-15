#pragma once

#include "gaffa/ascend_memory.h"
#include "gaffa/launch_ascend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gaffa {

struct AscendWeightedDownsamplePlan {
  std::size_t input_nsamples = 0;
  std::size_t output_nsamples = 0;
  double factor = 0.0;
  std::vector<std::uint32_t> first_input;
  std::vector<std::uint32_t> last_input;
  std::vector<float> first_weight;
  std::vector<float> last_weight;
};

AscendWeightedDownsamplePlan make_ascend_weighted_downsample_plan(
    std::size_t nsamples, double factor);

struct AscendTimeSeriesBatchView {
  const float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t nsamples = 0;
  int device_id = 0;

  [[nodiscard]] std::size_t size() const noexcept { return nseries * nsamples; }
};

struct MutableAscendTimeSeriesBatchView {
  float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t nsamples = 0;
  int device_id = 0;

  [[nodiscard]] std::size_t size() const noexcept { return nseries * nsamples; }

  [[nodiscard]] AscendTimeSeriesBatchView as_const() const noexcept {
    return AscendTimeSeriesBatchView{.data = data, .nseries = nseries,
                                     .nsamples = nsamples, .device_id = device_id};
  }
};

void downsample_weighted_sum_ascend(
    AscendTimeSeriesBatchView input, double factor, AscendSpan<float> output,
    const AscendLaunchOptions& options = {});

void convert_time_series_batch_to_float_ascend(
    AscendSpan<const std::uint32_t> input, std::size_t nseries,
    std::size_t nsamples, AscendSpan<float> output,
    const AscendLaunchOptions& options = {});

}  // namespace gaffa
