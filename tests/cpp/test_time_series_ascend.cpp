#include "gaffa/time_series.h"
#include "gaffa/time_series_ascend.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace {

std::vector<float> apply_plan(
    std::span<const float> input,
    const gaffa::AscendWeightedDownsamplePlan& plan) {
  std::vector<float> output(plan.output_nsamples);
  for (std::size_t index = 0; index < plan.output_nsamples; ++index) {
    float sum = plan.first_weight[index] * input[plan.first_input[index]];
    for (std::uint32_t input_index = plan.first_input[index] + 1;
         input_index < plan.last_input[index]; ++input_index) {
      sum += input[input_index];
    }
    sum += plan.last_weight[index] * input[plan.last_input[index]];
    output[index] = sum;
  }
  return output;
}

}  // namespace

TEST(TimeSeriesAscend, WeightedDownsamplePlanMatchesCpuReference) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7};

  for (const double factor : {2.0, 2.5, 3.25}) {
    const auto plan =
        gaffa::make_ascend_weighted_downsample_plan(input.size(), factor);
    ASSERT_EQ(plan.input_nsamples, input.size());
    ASSERT_EQ(plan.output_nsamples,
              gaffa::downsampled_size(input.size(), factor));
    ASSERT_EQ(plan.first_input.size(), plan.output_nsamples);
    ASSERT_EQ(plan.last_input.size(), plan.output_nsamples);
    ASSERT_EQ(plan.first_weight.size(), plan.output_nsamples);
    ASSERT_EQ(plan.last_weight.size(), plan.output_nsamples);

    const auto expected = gaffa::downsample_weighted_sum_cpu(input, factor);
    const auto actual = apply_plan(input, plan);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      EXPECT_NEAR(actual[index], expected[index], 1.0e-6F);
    }
  }
}
