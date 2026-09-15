#include "gaffa/ffa_ascend.h"

#include "gaffa/time_series.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace gaffa {
namespace {

std::size_t checked_mul_plan(std::size_t a, std::size_t b, const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) throw std::overflow_error(message);
  return a * b;
}
std::size_t checked_add_plan(std::size_t a, std::size_t b, const char* message) {
  if (b > std::numeric_limits<std::size_t>::max() - a) throw std::overflow_error(message);
  return a + b;
}
void validate_task(const FfaSearchTask& task) {
  if (task.input_nsamples == 0 || task.prepared_nsamples == 0 || task.bins <= 1 ||
      task.rows == 0 || task.rows_eval == 0 || task.rows_eval > task.rows ||
      !(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp) ||
      !std::isfinite(task.downsample_factor) || task.downsample_factor < 1.0)
    throw std::invalid_argument("Ascend FFA execution task is invalid");
  const std::size_t expected_prepared = task.downsample_factor == 1.0
      ? task.input_nsamples : downsampled_size(task.input_nsamples, task.downsample_factor);
  if (task.prepared_nsamples != expected_prepared)
    throw std::invalid_argument("Ascend FFA task prepared_nsamples must match downsample factor");
  const std::size_t elements = checked_mul_plan(task.rows, task.bins,
                                                 "Ascend FFA task element count overflow");
  if (elements > task.prepared_nsamples)
    throw std::invalid_argument("Ascend FFA task rows * bins exceeds prepared_nsamples");
}
bool same_prepare_key(const AscendFfaPrepareGroup& group, const FfaSearchTask& task) {
  return group.prepare_key.input_nsamples == task.input_nsamples &&
         group.prepare_key.downsample_factor == task.downsample_factor &&
         group.prepared_nsamples == task.prepared_nsamples;
}
}  // namespace

std::span<const AscendFfaPrepareGroup> AscendFfaExecutionPlan::groups() const noexcept { return groups_; }
std::size_t AscendFfaExecutionPlan::max_prepared_nsamples() const noexcept { return max_prepared_nsamples_; }
std::size_t AscendFfaExecutionPlan::max_transform_elements() const noexcept { return max_transform_elements_; }
std::size_t AscendFfaExecutionPlan::max_detection_slots_per_series() const noexcept { return max_detection_slots_per_series_; }

AscendFfaExecutionPlan make_ffa_ascend_execution_plan(const FfaSearchPlan& plan) {
  if (plan.tasks.empty()) throw std::invalid_argument("Ascend FFA plan must contain at least one task");
  if (plan.width_trials.empty()) throw std::invalid_argument("Ascend FFA plan width_trials must not be empty");
  AscendFfaExecutionPlan execution;
  for (const FfaSearchTask& task : plan.tasks) {
    validate_task(task);
    const FfaTransformShape shape{.rows = task.rows, .bins = task.bins};
    FfaDetectionPlan detection = make_ffa_detection_plan(plan.width_trials, task.bins);
    const std::size_t elements = checked_mul_plan(task.rows, task.bins,
                                                   "Ascend FFA task element count overflow");
    const std::size_t detection_slots = checked_mul_plan(
        task.rows_eval, detection.boxcar_trials.size(),
        "Ascend FFA detection slot count overflow");
    AscendFfaTaskLayout layout{
        .task = task, .shape = shape, .detection_plan = std::move(detection),
        .transform_schedule = compile_ascend_ffa_transform_schedule(shape),
        .transform_elements = elements,
        .detection_slots_per_series = detection_slots};
    auto group_it = std::find_if(execution.groups_.begin(), execution.groups_.end(),
                                 [&](const AscendFfaPrepareGroup& group) {
                                   return same_prepare_key(group, task);
                                 });
    if (group_it == execution.groups_.end()) {
      execution.groups_.push_back(AscendFfaPrepareGroup{
          .prepare_key = AscendFfaPrepareKey{.input_nsamples = task.input_nsamples,
                                             .downsample_factor = task.downsample_factor},
          .prepared_nsamples = task.prepared_nsamples, .tasks = {}});
      group_it = execution.groups_.end() - 1;
    }
    group_it->tasks.push_back(std::move(layout));
    execution.max_prepared_nsamples_ = std::max(execution.max_prepared_nsamples_, task.prepared_nsamples);
    execution.max_transform_elements_ = std::max(execution.max_transform_elements_, elements);
    execution.max_detection_slots_per_series_ = std::max(execution.max_detection_slots_per_series_, detection_slots);
  }
  return execution;
}

AscendFfaWorkspaceShape estimate_ffa_ascend_workspace(
    const AscendFfaExecutionPlan& plan, const AscendFfaExecutionOptions& options) {
  if (options.series_tile_size == 0) throw std::invalid_argument("Ascend FFA series_tile_size must be > 0");
  if (plan.groups().empty()) throw std::invalid_argument("Ascend FFA execution plan must not be empty");
  const std::size_t prepared_count = checked_mul_plan(options.series_tile_size, plan.max_prepared_nsamples(),
                                                       "Ascend FFA prepared workspace overflow");
  const std::size_t transform_count = checked_mul_plan(options.series_tile_size, plan.max_transform_elements(),
                                                        "Ascend FFA transform workspace overflow");
  const std::size_t detection_count = checked_mul_plan(options.series_tile_size, plan.max_detection_slots_per_series(),
                                                        "Ascend FFA detection workspace overflow");
  AscendFfaWorkspaceShape shape{
      .series_tile_size = options.series_tile_size,
      .max_prepared_nsamples = plan.max_prepared_nsamples(),
      .max_task_elements = plan.max_transform_elements(),
      .max_detection_slots_per_series = plan.max_detection_slots_per_series(),
      .prepared_bytes = checked_mul_plan(prepared_count, sizeof(float), "Ascend FFA prepared byte size overflow"),
      .scratch_bytes = checked_mul_plan(transform_count, sizeof(float), "Ascend FFA scratch byte size overflow"),
      .output_bytes = checked_mul_plan(transform_count, sizeof(float), "Ascend FFA output byte size overflow"),
      .detection_phase_bytes = checked_mul_plan(detection_count, sizeof(std::uint32_t), "Ascend FFA detection phase byte size overflow"),
      .detection_snr_bytes = checked_mul_plan(detection_count, sizeof(float), "Ascend FFA detection S/N byte size overflow")};
  shape.total_bytes = checked_add_plan(
      checked_add_plan(shape.prepared_bytes, shape.scratch_bytes, "Ascend FFA workspace byte size overflow"),
      checked_add_plan(shape.output_bytes,
          checked_add_plan(shape.detection_phase_bytes, shape.detection_snr_bytes, "Ascend FFA workspace byte size overflow"),
          "Ascend FFA workspace byte size overflow"),
      "Ascend FFA workspace byte size overflow");
  if (options.workspace_bytes_limit != 0 && shape.total_bytes > options.workspace_bytes_limit)
    throw std::runtime_error("Ascend FFA workspace exceeds workspace_bytes_limit");
  return shape;
}

}  // namespace gaffa
