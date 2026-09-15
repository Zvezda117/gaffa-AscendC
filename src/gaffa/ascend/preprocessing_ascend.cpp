#include "gaffa/preprocessing_ascend.h"

#include "gaffa/ascend_runtime.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" void gaffa_launch_validate_finite_ascend(
    void* data, void* status, std::uint32_t nseries, std::uint32_t nsamples,
    std::uint32_t block_dim, void* stream);
extern "C" void gaffa_launch_mean_scrunch_ascend(
    void* data, void* output, std::uint32_t nseries,
    std::uint32_t input_nsamples, std::uint32_t output_nsamples,
    std::uint32_t factor, std::uint32_t block_dim, void* stream);
extern "C" void gaffa_launch_running_median_ascend(
    void* input, void* baseline, std::uint32_t nseries,
    std::uint32_t input_nsamples, std::uint32_t output_nsamples,
    std::uint32_t window_samples, std::uint32_t block_dim, void* stream);
extern "C" void gaffa_launch_interpolate_subtract_ascend(
    void* data, void* baseline, std::uint32_t nseries,
    std::uint32_t nsamples, std::uint32_t baseline_nsamples,
    std::uint32_t factor, std::uint32_t block_dim, void* stream);
extern "C" void gaffa_launch_normalise_ascend(
    void* data, void* status, std::uint32_t nseries, std::uint32_t nsamples,
    int reject_constant, std::uint32_t block_dim, void* stream);

namespace gaffa {
namespace {

enum class AscendPreprocessStatus : int {
  Ok = 0,
  NonFiniteInput = 1,
  ConstantInput = 2,
};

void check_acl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::uint32_t checked_u32(std::size_t value, const char* message) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error(message);
  }
  return static_cast<std::uint32_t>(value);
}

std::uint32_t choose_block_dim(std::size_t nseries,
                               const AscendPreprocessExecutionOptions& options) {
  if (options.block_dim != 0) return options.block_dim;
  constexpr std::size_t kConservativeCommonCoreLimit = 24;
  return static_cast<std::uint32_t>(std::max<std::size_t>(
      1, std::min(nseries, kConservativeCommonCoreLimit)));
}

void validate_batch(const AscendPreprocessProgram& program,
                    MutableAscendTimeSeriesBatchView batch) {
  if (program.empty()) throw std::invalid_argument(
      "Ascend preprocessing program must not be empty");
  if (batch.data == nullptr) throw std::invalid_argument(
      "Ascend preprocessing batch data must not be null");
  if (batch.nseries == 0 || batch.nseries > program.tile_capacity())
    throw std::invalid_argument(
        "Ascend preprocessing batch nseries must fit program tile_capacity");
  if (batch.nsamples == 0 || batch.nsamples > program.max_nsamples())
    throw std::invalid_argument(
        "Ascend preprocessing batch nsamples must fit program max_nsamples");
  if (batch.device_id != program.device_id())
    throw std::invalid_argument(
        "Ascend preprocessing batch device_id must match program device_id");
}

}  // namespace

struct AscendPreprocessProgramImpl {
  enum class RunState { Idle, Pending, Poisoned };
  AscendPreprocessExecutionOptions execution_options;
  AscendPreprocessLayout layout;
  int device_id = 0;
  AscendStream stream = nullptr;
  std::unique_ptr<AscendRuntime> owned_runtime;
  AscendDeviceBuffer<float> scrunched;
  AscendDeviceBuffer<float> baseline;
  AscendDeviceBuffer<int> status;
  RunState run_state = RunState::Idle;
  std::size_t pending_nseries = 0;
};

namespace {

void validate_plan_for_batch(const AscendPreprocessProgramImpl& impl,
                             MutableAscendTimeSeriesBatchView batch) {
  for (const auto& step : impl.layout.steps) {
    if (step.kind != PreprocessStepKind::DetrendRunningMedian) continue;
    if (step.scrunch_factor == 1) {
      if (step.detrend_options.window_samples >= batch.nsamples)
        throw std::invalid_argument(
            "Ascend running median window_samples must be < input size");
    } else {
      const std::size_t reduced = batch.nsamples / step.scrunch_factor;
      if (reduced <= step.detrend_options.min_points)
        throw std::invalid_argument(
            "Ascend running median low-resolution sample count must be > min_points");
    }
  }
}

void run_detrend(AscendPreprocessProgramImpl& impl,
                 MutableAscendTimeSeriesBatchView batch,
                 const AscendPreprocessStepLayout& step) {
  const std::uint32_t nseries = checked_u32(
      batch.nseries, "Ascend preprocessing nseries exceeds uint32 range");
  const std::uint32_t nsamples = checked_u32(
      batch.nsamples, "Ascend preprocessing nsamples exceeds uint32 range");
  const std::uint32_t factor = checked_u32(
      step.scrunch_factor,
      "Ascend preprocessing scrunch factor exceeds uint32 range");
  const std::uint32_t block_dim = choose_block_dim(
      batch.nseries, impl.execution_options);

  void* median_input = batch.data;
  std::size_t median_nsamples = batch.nsamples;
  if (step.scrunch_factor > 1) {
    median_nsamples = batch.nsamples / step.scrunch_factor;
    gaffa_launch_mean_scrunch_ascend(
        batch.data, impl.scrunched.data(), nseries, nsamples,
        checked_u32(median_nsamples,
                    "Ascend preprocessing scrunched length exceeds uint32 range"),
        factor, block_dim, impl.stream);
    median_input = impl.scrunched.data();
  }

  gaffa_launch_running_median_ascend(
      median_input, impl.baseline.data(), nseries,
      checked_u32(median_nsamples,
                  "Ascend preprocessing median input length exceeds uint32 range"),
      checked_u32(median_nsamples,
                  "Ascend preprocessing median output length exceeds uint32 range"),
      checked_u32(step.median_window,
                  "Ascend preprocessing median window exceeds uint32 range"),
      block_dim, impl.stream);

  gaffa_launch_interpolate_subtract_ascend(
      batch.data, impl.baseline.data(), nseries, nsamples,
      checked_u32(median_nsamples,
                  "Ascend preprocessing baseline length exceeds uint32 range"),
      factor, block_dim, impl.stream);
}

void run_normalise(AscendPreprocessProgramImpl& impl,
                   MutableAscendTimeSeriesBatchView batch,
                   NormaliseOptions options) {
  gaffa_launch_normalise_ascend(
      batch.data, impl.status.data(),
      checked_u32(batch.nseries,
                  "Ascend preprocessing nseries exceeds uint32 range"),
      checked_u32(batch.nsamples,
                  "Ascend preprocessing nsamples exceeds uint32 range"),
      options.reject_constant ? 1 : 0,
      choose_block_dim(batch.nseries, impl.execution_options), impl.stream);
}

}  // namespace

AscendPreprocessProgram::AscendPreprocessProgram(
    PreprocessPlan plan, const AscendPreprocessProgramOptions& program_options,
    const AscendPreprocessExecutionOptions& execution_options)
    : impl_(std::make_unique<AscendPreprocessProgramImpl>()) {
  if (program_options.device_id < 0)
    throw std::invalid_argument(
        "Ascend preprocessing program device_id must be >= 0");
  impl_->execution_options = execution_options;
  impl_->layout = compile_ascend_preprocess_layout(plan, execution_options);
  impl_->device_id = program_options.device_id;

  if (execution_options.stream == nullptr) {
    impl_->owned_runtime = std::make_unique<AscendRuntime>(program_options.device_id);
    impl_->stream = impl_->owned_runtime->stream();
  } else {
    check_acl(aclrtSetDevice(program_options.device_id),
              "aclrtSetDevice for Ascend preprocessing program");
    impl_->stream = execution_options.stream;
  }

  impl_->scrunched = AscendDeviceBuffer<float>(
      impl_->layout.scrunched_count, impl_->device_id);
  impl_->baseline = AscendDeviceBuffer<float>(
      impl_->layout.baseline_count, impl_->device_id);
  impl_->status = AscendDeviceBuffer<int>(
      execution_options.series_tile_size, impl_->device_id);
}

AscendPreprocessProgram::~AscendPreprocessProgram() = default;
AscendPreprocessProgram::AscendPreprocessProgram(
    AscendPreprocessProgram&&) noexcept = default;
AscendPreprocessProgram& AscendPreprocessProgram::operator=(
    AscendPreprocessProgram&&) noexcept = default;

bool AscendPreprocessProgram::empty() const noexcept { return impl_ == nullptr; }

int AscendPreprocessProgram::device_id() const {
  if (empty()) throw std::logic_error(
      "Ascend preprocessing program must not be empty");
  return impl_->device_id;
}

std::size_t AscendPreprocessProgram::tile_capacity() const {
  if (empty()) throw std::logic_error(
      "Ascend preprocessing program must not be empty");
  return impl_->execution_options.series_tile_size;
}

std::size_t AscendPreprocessProgram::max_nsamples() const {
  if (empty()) throw std::logic_error(
      "Ascend preprocessing program must not be empty");
  return impl_->execution_options.max_nsamples;
}

const AscendPreprocessWorkspaceShape&
AscendPreprocessProgram::workspace_shape() const {
  if (empty()) throw std::logic_error(
      "Ascend preprocessing program must not be empty");
  return impl_->layout.workspace;
}

void AscendPreprocessProgram::synchronize() {
  if (empty()) throw std::logic_error(
      "Ascend preprocessing program must not be empty");
  if (impl_->run_state == AscendPreprocessProgramImpl::RunState::Poisoned)
    throw std::logic_error(
        "Ascend preprocessing program is poisoned by an earlier runtime failure; recreate it");
  if (impl_->run_state == AscendPreprocessProgramImpl::RunState::Idle) return;

  std::vector<int> status(impl_->pending_nseries, 0);
  try {
    check_acl(aclrtSetDevice(impl_->device_id),
              "aclrtSetDevice for Ascend preprocessing synchronize");
    check_acl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(impl_->stream)),
              "aclrtSynchronizeStream for Ascend preprocessing");
    check_acl(aclrtMemcpy(status.data(), status.size() * sizeof(int),
                          impl_->status.data(), status.size() * sizeof(int),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "Ascend preprocessing status D2H");
  } catch (...) {
    impl_->run_state = AscendPreprocessProgramImpl::RunState::Poisoned;
    throw;
  }
  impl_->run_state = AscendPreprocessProgramImpl::RunState::Idle;
  impl_->pending_nseries = 0;

  for (const int raw_status : status) {
    switch (static_cast<AscendPreprocessStatus>(raw_status)) {
      case AscendPreprocessStatus::Ok:
        break;
      case AscendPreprocessStatus::NonFiniteInput:
        throw std::invalid_argument(
            "Ascend preprocessing input samples must be finite");
      case AscendPreprocessStatus::ConstantInput:
        throw std::invalid_argument(
            "Ascend normalise input standard deviation must be finite and > 0");
      default:
        throw std::logic_error("Ascend preprocessing status is invalid");
    }
  }
}

void preprocess_time_series_batch_inplace_ascend(
    AscendPreprocessProgram& program, MutableAscendTimeSeriesBatchView batch) {
  validate_batch(program, batch);
  auto& impl = *program.impl_;
  if (impl.run_state == AscendPreprocessProgramImpl::RunState::Poisoned)
    throw std::logic_error(
        "Ascend preprocessing program is poisoned by an earlier runtime failure; recreate it");
  if (impl.run_state == AscendPreprocessProgramImpl::RunState::Pending)
    throw std::logic_error(
        "Ascend preprocessing program already has an active run; call synchronize() before reuse");
  validate_plan_for_batch(impl, batch);

  check_acl(aclrtSetDevice(impl.device_id),
            "aclrtSetDevice for Ascend preprocessing run");
  impl.run_state = AscendPreprocessProgramImpl::RunState::Pending;
  impl.pending_nseries = batch.nseries;
  try {
    std::vector<int> zero_status(batch.nseries, 0);
    check_acl(aclrtMemcpy(impl.status.data(), impl.status.bytes(),
                          zero_status.data(), zero_status.size() * sizeof(int),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "Ascend preprocessing status reset");

    const std::uint32_t block_dim = choose_block_dim(
        batch.nseries, impl.execution_options);
    gaffa_launch_validate_finite_ascend(
        batch.data, impl.status.data(),
        checked_u32(batch.nseries,
                    "Ascend preprocessing nseries exceeds uint32 range"),
        checked_u32(batch.nsamples,
                    "Ascend preprocessing nsamples exceeds uint32 range"),
        block_dim, impl.stream);

    for (const auto& step : impl.layout.steps) {
      switch (step.kind) {
        case PreprocessStepKind::DetrendRunningMedian:
          run_detrend(impl, batch, step);
          break;
        case PreprocessStepKind::Normalise:
          run_normalise(impl, batch, step.normalise);
          break;
      }
    }
  } catch (...) {
    impl.run_state = AscendPreprocessProgramImpl::RunState::Poisoned;
    throw;
  }
}

}  // namespace gaffa
