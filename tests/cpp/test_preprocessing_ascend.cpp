#include "gaffa/preprocessing_ascend.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(PreprocessingAscend, CompilesRunningMedianWorkspace) {
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
          .detrend_running_median = {.window_samples = 101, .min_points = 5},
      }},
  };
  const gaffa::AscendPreprocessExecutionOptions options{
      .series_tile_size = 2,
      .max_nsamples = 1000,
  };

  const auto layout = gaffa::compile_ascend_preprocess_layout(plan, options);

  ASSERT_EQ(layout.steps.size(), 1U);
  EXPECT_EQ(layout.steps[0].scrunch_factor, 20U);
  EXPECT_EQ(layout.steps[0].median_window, 5U);
  EXPECT_EQ(layout.workspace.scrunched_bytes,
            2U * (1000U / 20U) * sizeof(float));
  EXPECT_EQ(layout.workspace.baseline_bytes,
            layout.workspace.scrunched_bytes);
  EXPECT_EQ(layout.workspace.total_bytes,
            layout.workspace.scrunched_bytes +
                layout.workspace.baseline_bytes +
                layout.workspace.partial_stats_bytes +
                layout.workspace.series_stats_bytes +
                layout.workspace.status_bytes);
}

TEST(PreprocessingAscend, NormaliseOnlyUsesPerSeriesStatus) {
  const gaffa::PreprocessPlan plan{
      .steps = {
          gaffa::PreprocessStep{.kind = gaffa::PreprocessStepKind::Normalise},
      },
  };

  const auto layout = gaffa::compile_ascend_preprocess_layout(
      plan, {.series_tile_size = 3, .max_nsamples = 4097});

  EXPECT_EQ(layout.workspace.partial_stats_bytes, 0U);
  EXPECT_EQ(layout.workspace.series_stats_bytes, 0U);
  EXPECT_EQ(layout.workspace.status_bytes, 3U * sizeof(int));
}

TEST(PreprocessingAscend, RejectsWorkspaceOverflowEmptyPlanAndOversizedMedian) {
  const gaffa::PreprocessPlan plan{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
          .detrend_running_median = {.window_samples = 101, .min_points = 5},
      }},
  };
  const gaffa::AscendPreprocessExecutionOptions options{
      .series_tile_size = 2,
      .max_nsamples = 1000,
  };
  const auto layout = gaffa::compile_ascend_preprocess_layout(plan, options);

  EXPECT_THROW(
      (void)gaffa::compile_ascend_preprocess_layout(
          plan,
          {.series_tile_size = 2,
           .max_nsamples = 1000,
           .workspace_bytes_limit = layout.workspace.total_bytes - 1}),
      std::runtime_error);

  EXPECT_THROW((void)gaffa::compile_ascend_preprocess_layout({}, options),
               std::invalid_argument);

  const gaffa::PreprocessPlan oversized{
      .steps = {gaffa::PreprocessStep{
          .kind = gaffa::PreprocessStepKind::DetrendRunningMedian,
          .detrend_running_median = {.window_samples = 513,
                                     .min_points = 513},
      }},
  };
  EXPECT_THROW(
      (void)gaffa::compile_ascend_preprocess_layout(oversized, options),
      std::invalid_argument);
}
