#include "gaffa/dedispersion_ascend.h"
#include "gaffa/internal/dedispersion_delay.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(DedispersionAscend, SingleDmLayoutMatchesCpuDelayDefinition) {
  const std::vector<double> frequency{1400.0, 1300.0, 1200.0, 1100.0};
  const gaffa::SingleDmDedispersionPlan plan{
      .dm = 50.0,
      .ref_frequency_mhz = 1400.0,
      .tsamp = 0.001,
      .chan_begin = 0,
      .chan_end = 4,
  };

  const auto layout = gaffa::compile_ascend_single_dm_layout(
      gaffa::SampleShape{.nsamples = 4096, .nifs = 1, .nchans = 4},
      frequency, plan);
  const auto expected = gaffa::internal::make_single_dm_delay_table(
      frequency, plan.dm, plan.ref_frequency_mhz, plan.tsamp,
      plan.chan_begin, plan.chan_end);

  EXPECT_EQ(layout.delays, expected);
  EXPECT_EQ(layout.output_shape.ndm, 1U);
  EXPECT_EQ(layout.output_shape.nsamples,
            gaffa::internal::valid_output_nsamples(
                4096,
                gaffa::internal::single_dm_max_delay(frequency, plan)));
  EXPECT_EQ(layout.channel_count, 4U);
  EXPECT_EQ(layout.chan_begin, 0U);
}

TEST(DedispersionAscend, MultiDmLayoutMatchesCpuDelayDefinition) {
  const std::vector<double> frequency{1400.0, 1300.0, 1200.0, 1100.0};
  const gaffa::MultiDmDedispersionPlan plan{
      .dm_low = 10.0,
      .dm_step = 2.5,
      .ndm = 4,
      .ref_frequency_mhz = 1400.0,
      .tsamp = 0.001,
      .chan_begin = 1,
      .chan_end = 4,
  };

  const auto layout = gaffa::compile_ascend_multi_dm_layout(
      gaffa::SampleShape{.nsamples = 4096, .nifs = 1, .nchans = 4},
      frequency, plan);

  EXPECT_EQ(layout.delays,
            gaffa::internal::make_multi_dm_delay_table(frequency, plan));
  EXPECT_EQ(layout.output_shape.ndm, 4U);
  EXPECT_EQ(layout.channel_count, 3U);
  EXPECT_EQ(layout.chan_begin, 1U);
  EXPECT_EQ(layout.output_shape.nsamples,
            gaffa::internal::valid_output_nsamples(
                4096,
                gaffa::internal::multi_dm_max_delay(frequency, plan)));
}

TEST(DedispersionAscend, RejectsInvalidFrequencyRangeAndNifs) {
  const std::vector<double> bad_frequency{1400.0, 1500.0};
  EXPECT_THROW(
      (void)gaffa::compile_ascend_single_dm_layout(
          {.nsamples = 64, .nifs = 1, .nchans = 2}, bad_frequency,
          {.dm = 1.0,
           .ref_frequency_mhz = 1400.0,
           .tsamp = 0.001,
           .chan_begin = 0,
           .chan_end = 2}),
      std::invalid_argument);

  const std::vector<double> frequency{1400.0, 1300.0, 1200.0, 1100.0};
  EXPECT_THROW(
      (void)gaffa::compile_ascend_multi_dm_layout(
          {.nsamples = 64, .nifs = 2, .nchans = 4}, frequency,
          {.dm_low = 0.0,
           .dm_step = 1.0,
           .ndm = 2,
           .ref_frequency_mhz = 1400.0,
           .tsamp = 0.001,
           .chan_begin = 0,
           .chan_end = 4}),
      std::invalid_argument);
}
