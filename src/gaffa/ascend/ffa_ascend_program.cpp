#include "gaffa/ffa_ascend.h"

#include <acl/acl.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace gaffa {

struct AscendFfaProgramImpl {
  AscendFfaExecutionPlan execution_plan;
  AscendFfaExecutionOptions execution_options;
  AscendFfaWorkspaceShape workspace_shape;
  int device_id = 0;
  AscendDeviceBuffer<float> prepared;
  AscendDeviceBuffer<float> scratch;
  AscendDeviceBuffer<float> output;
};

namespace {
void check_acl_program(aclError status, const char* operation) {
  if (status != ACL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
}
std::size_t checked_mul_program(std::size_t a, std::size_t b, const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) throw std::overflow_error(message);
  return a * b;
}
AscendLaunchOptions program_launch_options(const AscendFfaProgramImpl& impl) {
  return AscendLaunchOptions{.device_id = impl.device_id,
                             .block_dim = impl.execution_options.block_dim,
                             .stream = impl.execution_options.stream,
                             .synchronize_after_call = true};
}
}  // namespace

AscendFfaProgram::AscendFfaProgram(
    AscendFfaExecutionPlan execution_plan,
    const AscendFfaProgramOptions& program_options,
    const AscendFfaExecutionOptions& execution_options)
    : impl_(std::make_unique<AscendFfaProgramImpl>()) {
  if (program_options.device_id < 0) throw std::invalid_argument("Ascend FFA program device_id must be >= 0");
  impl_->execution_plan = std::move(execution_plan);
  impl_->execution_options = execution_options;
  impl_->workspace_shape = estimate_ffa_ascend_workspace(impl_->execution_plan, execution_options);
  impl_->device_id = program_options.device_id;
  check_acl_program(aclrtSetDevice(program_options.device_id), "aclrtSetDevice for Ascend FFA program");
  const std::size_t prepared_count = checked_mul_program(
      impl_->workspace_shape.series_tile_size, impl_->workspace_shape.max_prepared_nsamples,
      "Ascend FFA prepared allocation count overflow");
  const std::size_t transform_count = checked_mul_program(
      impl_->workspace_shape.series_tile_size, impl_->workspace_shape.max_task_elements,
      "Ascend FFA transform allocation count overflow");
  impl_->prepared = AscendDeviceBuffer<float>(prepared_count, impl_->device_id);
  impl_->scratch = AscendDeviceBuffer<float>(transform_count, impl_->device_id);
  impl_->output = AscendDeviceBuffer<float>(transform_count, impl_->device_id);
}

AscendFfaProgram::AscendFfaProgram(
    const FfaSearchPlan& plan, const AscendFfaProgramOptions& program_options,
    const AscendFfaExecutionOptions& execution_options)
    : AscendFfaProgram(make_ffa_ascend_execution_plan(plan), program_options, execution_options) {}
AscendFfaProgram::~AscendFfaProgram() = default;
AscendFfaProgram::AscendFfaProgram(AscendFfaProgram&&) noexcept = default;
AscendFfaProgram& AscendFfaProgram::operator=(AscendFfaProgram&&) noexcept = default;
bool AscendFfaProgram::empty() const noexcept { return impl_ == nullptr; }
const AscendFfaExecutionPlan& AscendFfaProgram::execution_plan() const {
  if (empty()) throw std::logic_error("Ascend FFA program must not be empty"); return impl_->execution_plan;
}
int AscendFfaProgram::device_id() const {
  if (empty()) throw std::logic_error("Ascend FFA program must not be empty"); return impl_->device_id;
}
std::size_t AscendFfaProgram::tile_capacity() const {
  if (empty()) throw std::logic_error("Ascend FFA program must not be empty"); return impl_->workspace_shape.series_tile_size;
}
const AscendFfaWorkspaceShape& AscendFfaProgram::workspace_shape() const {
  if (empty()) throw std::logic_error("Ascend FFA program must not be empty"); return impl_->workspace_shape;
}

AscendFfaBatchSearchResult run_ffa_batch_ascend(
    AscendFfaProgram& program, AscendTimeSeriesBatchView batch,
    const FfaSearchOptions& options) {
  if (program.empty()) throw std::invalid_argument("Ascend FFA program must not be empty");
  AscendFfaProgramImpl& impl = *program.impl_;
  if (batch.data == nullptr || batch.nseries == 0 || batch.nsamples == 0)
    throw std::invalid_argument("Ascend FFA batch must be non-empty");
  if (batch.nseries > impl.workspace_shape.series_tile_size)
    throw std::invalid_argument("Ascend FFA batch nseries exceeds program tile capacity");
  if (batch.device_id != impl.device_id)
    throw std::invalid_argument("Ascend FFA batch device_id must match program device_id");
  const AscendLaunchOptions launch = program_launch_options(impl);
  AscendFfaBatchSearchResult result;
  for (const AscendFfaPrepareGroup& group : impl.execution_plan.groups()) {
    if (group.prepare_key.input_nsamples != batch.nsamples || group.tasks.empty())
      throw std::invalid_argument("Ascend FFA execution group does not match input batch");
    const FfaSearchTask& prepare_task = group.tasks.front().task;
    const std::size_t prepared_count = checked_mul_program(batch.nseries, group.prepared_nsamples,
                                                            "Ascend FFA prepared batch count overflow");
    prepare_ffa_input_ascend(batch, prepare_task,
        AscendSpan<float>{.data = impl.prepared.data(), .count = prepared_count,
                          .device_id = impl.device_id}, launch);
    for (const AscendFfaTaskLayout& task_layout : group.tasks) {
      const std::size_t stride = impl.workspace_shape.max_task_elements;
      ffa_transform_block_ascend(
          AscendFfaInput{.data = impl.prepared.data(), .nseries = batch.nseries,
                         .nsamples = group.prepared_nsamples, .stride = group.prepared_nsamples,
                         .shape = task_layout.shape, .device_id = impl.device_id},
          AscendFfaBuffer{.data = impl.scratch.data(), .nseries = batch.nseries,
                          .stride = stride, .shape = task_layout.shape, .device_id = impl.device_id},
          AscendFfaBuffer{.data = impl.output.data(), .nseries = batch.nseries,
                          .stride = stride, .shape = task_layout.shape, .device_id = impl.device_id}, launch);
      std::vector<std::size_t> widths;
      widths.reserve(task_layout.detection_plan.boxcar_trials.size());
      for (const auto& trial : task_layout.detection_plan.boxcar_trials) widths.push_back(trial.width);
      AscendFfaBatchSearchResult task_result = detect_ffa_batch_ascend(
          AscendFfaBuffer{.data = impl.output.data(), .nseries = batch.nseries,
                          .stride = stride, .shape = task_layout.shape, .device_id = impl.device_id},
          task_layout.task, widths, options, launch);
      result.peaks.insert(result.peaks.end(), task_result.peaks.begin(), task_result.peaks.end());
    }
  }
  return result;
}

}  // namespace gaffa
