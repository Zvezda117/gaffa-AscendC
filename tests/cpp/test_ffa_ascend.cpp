#include "gaffa/ffa_ascend.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

TEST(FfaAscend, CompilesBaseAndRecursiveTransformSchedules) {
  {
    const auto schedule =
        gaffa::compile_ascend_ffa_transform_schedule({1, 8});
    ASSERT_EQ(schedule.levels.size(), 1U);
    ASSERT_EQ(schedule.levels[0].copy_ops.size(), 1U);
    EXPECT_TRUE(schedule.levels[0].merge_ops.empty());
    const auto& op = schedule.levels[0].copy_ops.front();
    EXPECT_EQ(op.input_role, gaffa::AscendFfaBufferRole::Input);
    EXPECT_EQ(op.output_role, gaffa::AscendFfaBufferRole::Output);
    EXPECT_EQ(op.rows, 1U);
  }

  {
    const auto schedule =
        gaffa::compile_ascend_ffa_transform_schedule({2, 8});
    ASSERT_EQ(schedule.levels.size(), 1U);
    EXPECT_TRUE(schedule.levels[0].copy_ops.empty());
    ASSERT_EQ(schedule.levels[0].merge_ops.size(), 1U);
    EXPECT_EQ(schedule.levels[0].merge_ops[0].output_rows, 2U);
  }

  {
    const auto schedule =
        gaffa::compile_ascend_ffa_transform_schedule({5, 16});
    ASSERT_EQ(schedule.levels.size(), 3U);
    ASSERT_EQ(schedule.levels.back().merge_ops.size(), 1U);
    const auto& root = schedule.levels.back().merge_ops.front();
    EXPECT_EQ(root.head_rows, 2U);
    EXPECT_EQ(root.tail_rows, 3U);
    EXPECT_EQ(root.output_rows, 5U);
    EXPECT_EQ(root.output_role, gaffa::AscendFfaBufferRole::Output);
  }
}

TEST(FfaAscend, RejectsInvalidTransformShapes) {
  EXPECT_THROW(
      (void)gaffa::compile_ascend_ffa_transform_schedule({0, 8}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)gaffa::compile_ascend_ffa_transform_schedule({4, 1}),
      std::invalid_argument);
}

TEST(FfaAscend, GroupsPrepareTasksAndSizesReusableWorkspace) {
  const gaffa::FfaSearchPlan plan{
      .tasks = {
          gaffa::FfaSearchTask{
              .downsample_factor = 1.0,
              .effective_tsamp = 0.001,
              .input_nsamples = 100,
              .prepared_nsamples = 100,
              .bins = 10,
              .rows = 10,
              .rows_eval = 8,
          },
          gaffa::FfaSearchTask{
              .downsample_factor = 1.0,
              .effective_tsamp = 0.001,
              .input_nsamples = 100,
              .prepared_nsamples = 100,
              .bins = 20,
              .rows = 5,
              .rows_eval = 5,
          },
          gaffa::FfaSearchTask{
              .downsample_factor = 2.0,
              .effective_tsamp = 0.002,
              .input_nsamples = 100,
              .prepared_nsamples = 50,
              .bins = 10,
              .rows = 5,
              .rows_eval = 4,
          },
      },
      .width_trials = {1, 2, 3},
  };

  const auto execution = gaffa::make_ffa_ascend_execution_plan(plan);
  ASSERT_EQ(execution.groups().size(), 2U);
  EXPECT_EQ(execution.groups()[0].tasks.size(), 2U);
  EXPECT_EQ(execution.groups()[1].tasks.size(), 1U);
  EXPECT_EQ(execution.max_prepared_nsamples(), 100U);
  EXPECT_EQ(execution.max_transform_elements(), 100U);
  EXPECT_EQ(execution.max_detection_slots_per_series(), 8U * 3U);

  const gaffa::AscendFfaExecutionOptions options{.series_tile_size = 4};
  const auto shape = gaffa::estimate_ffa_ascend_workspace(execution, options);
  EXPECT_EQ(shape.series_tile_size, 4U);
  EXPECT_EQ(shape.prepared_bytes, 4U * 100U * sizeof(float));
  EXPECT_EQ(shape.scratch_bytes, 4U * 100U * sizeof(float));
  EXPECT_EQ(shape.output_bytes, 4U * 100U * sizeof(float));
  EXPECT_EQ(shape.detection_phase_bytes,
            4U * 24U * sizeof(std::uint32_t));
  EXPECT_EQ(shape.detection_snr_bytes, 4U * 24U * sizeof(float));
  EXPECT_EQ(shape.total_bytes,
            shape.prepared_bytes + shape.scratch_bytes + shape.output_bytes +
                shape.detection_phase_bytes + shape.detection_snr_bytes);

  EXPECT_THROW(
      (void)gaffa::estimate_ffa_ascend_workspace(
          execution,
          gaffa::AscendFfaExecutionOptions{
              .series_tile_size = 4,
              .workspace_bytes_limit = shape.total_bytes - 1,
          }),
      std::runtime_error);
}
