#include "gaffa/time_series_ascend.h"

#include "gaffa/time_series.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gaffa {

AscendWeightedDownsamplePlan make_ascend_weighted_downsample_plan(
    std::size_t nsamples, double factor) {
  const std::size_t output_nsamples = downsampled_size(nsamples, factor);
  if (nsamples - 1 > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error(
        "Ascend weighted downsample supports at most UINT32_MAX + 1 samples");
  }

  AscendWeightedDownsamplePlan plan;
  plan.input_nsamples = nsamples;
  plan.output_nsamples = output_nsamples;
  plan.factor = factor;
  plan.first_input.resize(output_nsamples);
  plan.last_input.resize(output_nsamples);
  plan.first_weight.resize(output_nsamples);
  plan.last_weight.resize(output_nsamples);

  const double last_input_index = static_cast<double>(nsamples - 1);
  for (std::size_t output_index = 0; output_index < output_nsamples; ++output_index) {
    const double start = static_cast<double>(output_index) * factor;
    const double end = start + factor;
    const auto first_input = static_cast<std::size_t>(std::floor(start));
    const auto last_input = static_cast<std::size_t>(
        std::min(std::floor(end), last_input_index));
    plan.first_input[output_index] = static_cast<std::uint32_t>(first_input);
    plan.last_input[output_index] = static_cast<std::uint32_t>(last_input);
    plan.first_weight[output_index] = static_cast<float>(
        static_cast<double>(first_input + 1) - start);
    plan.last_weight[output_index] =
        static_cast<float>(end - static_cast<double>(last_input));
  }
  return plan;
}

}  // namespace gaffa
