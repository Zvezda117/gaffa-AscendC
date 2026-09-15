#pragma once

#include "gaffa/ascend_memory.h"
#include "gaffa/ffa.h"
#include "gaffa/ffa_detection.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"
#include "gaffa/launch_ascend.h"
#include "gaffa/time_series_ascend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace gaffa {

enum class AscendFfaBufferRole : std::uint32_t { Input = 0, Scratch = 1, Output = 2 };

struct AscendFfaCopyOp {
  AscendFfaBufferRole input_role = AscendFfaBufferRole::Input;
  AscendFfaBufferRole output_role = AscendFfaBufferRole::Output;
  std::uint32_t input_begin_row = 0;
  std::uint32_t output_begin_row = 0;
  std::uint32_t rows = 0;
};

struct AscendFfaMergeOp {
  AscendFfaBufferRole head_role = AscendFfaBufferRole::Input;
  AscendFfaBufferRole tail_role = AscendFfaBufferRole::Input;
  AscendFfaBufferRole output_role = AscendFfaBufferRole::Output;
  std::uint32_t head_begin_row = 0;
  std::uint32_t tail_begin_row = 0;
  std::uint32_t output_begin_row = 0;
  std::uint32_t head_rows = 0;
  std::uint32_t tail_rows = 0;
  std::uint32_t output_rows = 0;
};

struct AscendFfaTransformLevel {
  std::vector<AscendFfaCopyOp> copy_ops;
  std::vector<AscendFfaMergeOp> merge_ops;
};

struct AscendFfaTransformSchedule {
  FfaTransformShape shape{};
  std::vector<AscendFfaTransformLevel> levels;
};

AscendFfaTransformSchedule compile_ascend_ffa_transform_schedule(FfaTransformShape shape);

struct AscendFfaProgramOptions { int device_id = 0; };

struct AscendFfaExecutionOptions {
  std::size_t series_tile_size = 16;
  std::size_t workspace_bytes_limit = 0;
  std::uint32_t block_dim = 0;
  AscendStream stream = nullptr;
};

struct AscendFfaWorkspaceShape {
  std::size_t series_tile_size = 0;
  std::size_t max_prepared_nsamples = 0;
  std::size_t max_task_elements = 0;
  std::size_t max_detection_slots_per_series = 0;
  std::size_t prepared_bytes = 0;
  std::size_t scratch_bytes = 0;
  std::size_t output_bytes = 0;
  std::size_t detection_phase_bytes = 0;
  std::size_t detection_snr_bytes = 0;
  std::size_t total_bytes = 0;
};

struct AscendFfaPrepareKey {
  std::size_t input_nsamples = 0;
  double downsample_factor = 1.0;
};

struct AscendFfaTaskLayout {
  FfaSearchTask task{};
  FfaTransformShape shape{};
  FfaDetectionPlan detection_plan{};
  AscendFfaTransformSchedule transform_schedule{};
  std::size_t transform_elements = 0;
  std::size_t detection_slots_per_series = 0;
};

struct AscendFfaPrepareGroup {
  AscendFfaPrepareKey prepare_key{};
  std::size_t prepared_nsamples = 0;
  std::vector<AscendFfaTaskLayout> tasks;
};

class AscendFfaExecutionPlan {
 public:
  AscendFfaExecutionPlan() = default;
  [[nodiscard]] std::span<const AscendFfaPrepareGroup> groups() const noexcept;
  [[nodiscard]] std::size_t max_prepared_nsamples() const noexcept;
  [[nodiscard]] std::size_t max_transform_elements() const noexcept;
  [[nodiscard]] std::size_t max_detection_slots_per_series() const noexcept;
 private:
  friend AscendFfaExecutionPlan make_ffa_ascend_execution_plan(const FfaSearchPlan& plan);
  std::vector<AscendFfaPrepareGroup> groups_;
  std::size_t max_prepared_nsamples_ = 0;
  std::size_t max_transform_elements_ = 0;
  std::size_t max_detection_slots_per_series_ = 0;
};

AscendFfaExecutionPlan make_ffa_ascend_execution_plan(const FfaSearchPlan& plan);
AscendFfaWorkspaceShape estimate_ffa_ascend_workspace(
    const AscendFfaExecutionPlan& plan,
    const AscendFfaExecutionOptions& options = {});

struct AscendFfaProgramImpl;
struct AscendFfaBatchSearchResult;

class AscendFfaProgram {
 public:
  explicit AscendFfaProgram(AscendFfaExecutionPlan execution_plan,
                            const AscendFfaProgramOptions& program_options = {},
                            const AscendFfaExecutionOptions& execution_options = {});
  explicit AscendFfaProgram(const FfaSearchPlan& plan,
                            const AscendFfaProgramOptions& program_options = {},
                            const AscendFfaExecutionOptions& execution_options = {});
  ~AscendFfaProgram();
  AscendFfaProgram(AscendFfaProgram&&) noexcept;
  AscendFfaProgram& operator=(AscendFfaProgram&&) noexcept;
  AscendFfaProgram(const AscendFfaProgram&) = delete;
  AscendFfaProgram& operator=(const AscendFfaProgram&) = delete;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] const AscendFfaExecutionPlan& execution_plan() const;
  [[nodiscard]] int device_id() const;
  [[nodiscard]] std::size_t tile_capacity() const;
  [[nodiscard]] const AscendFfaWorkspaceShape& workspace_shape() const;
 private:
  friend AscendFfaBatchSearchResult run_ffa_batch_ascend(
      AscendFfaProgram& program, AscendTimeSeriesBatchView batch,
      const FfaSearchOptions& options);
  std::unique_ptr<AscendFfaProgramImpl> impl_;
};

AscendFfaBatchSearchResult run_ffa_batch_ascend(
    AscendFfaProgram& program, AscendTimeSeriesBatchView batch,
    const FfaSearchOptions& options = {});

struct AscendFfaInput {
  const float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t nsamples = 0;
  std::size_t stride = 0;
  FfaTransformShape shape{};
  int device_id = 0;
  [[nodiscard]] std::size_t task_elements() const noexcept { return shape.rows * shape.bins; }
};

struct AscendFfaBuffer {
  float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t stride = 0;
  FfaTransformShape shape{};
  int device_id = 0;
  [[nodiscard]] std::size_t task_elements() const noexcept { return shape.rows * shape.bins; }
};

void prepare_ffa_input_ascend(
    AscendTimeSeriesBatchView input, const FfaSearchTask& task,
    AscendSpan<float> output, const AscendLaunchOptions& options = {});

void ffa_transform_block_ascend(
    AscendFfaInput input, AscendFfaBuffer scratch, AscendFfaBuffer output,
    const AscendLaunchOptions& options = {});

struct AscendFfaBatchPeak {
  std::size_t series_index = 0;
  FfaPeak peak{};
};

struct AscendFfaBatchSearchResult { std::vector<AscendFfaBatchPeak> peaks; };

AscendFfaBatchSearchResult detect_ffa_batch_ascend(
    AscendFfaBuffer transform, const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    const FfaSearchOptions& search_options = {},
    const AscendLaunchOptions& launch_options = {});

AscendFfaBatchSearchResult search_ffa_batch_ascend(
    AscendTimeSeriesBatchView input, const FfaSearchPlan& plan,
    const FfaSearchOptions& search_options = {},
    const AscendLaunchOptions& launch_options = {});

}  // namespace gaffa
