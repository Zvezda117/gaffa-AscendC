#include "gaffa/ffa_ascend.h"

#include "gaffa/ffa_detection.h"

#include <acl/acl.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void gaffa_launch_ffa_detect_ascend(
    void* transform, std::uint32_t transform_stride, std::uint32_t rows_eval,
    std::uint32_t bins, void* widths, void* heights, void* baselines,
    std::uint32_t trial_count, float stdnoise, void* phases, void* snrs,
    std::uint32_t nseries, std::uint32_t block_dim, void* stream);

namespace gaffa {
namespace {

void check_acl_detection(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::size_t checked_mul_detection(std::size_t a, std::size_t b,
                                  const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error(message);
  return a * b;
}

std::uint32_t checked_u32(std::size_t value, const char* message) {
  if (value > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error(message);
  return static_cast<std::uint32_t>(value);
}

void validate_detection_input(AscendFfaBuffer transform,
                              const FfaSearchTask& task,
                              const FfaSearchOptions& search_options,
                              const AscendLaunchOptions& launch_options) {
  if (transform.data == nullptr || transform.nseries == 0)
    throw std::invalid_argument("Ascend FFA detection transform must be non-empty");
  if (launch_options.device_id < 0 || transform.device_id != launch_options.device_id)
    throw std::invalid_argument("Ascend FFA detection device ids must match launch device_id");
  if (!launch_options.synchronize_after_call)
    throw std::invalid_argument("Ascend FFA detection requires synchronized launch");
  if (task.rows == 0 || task.bins <= 1 || task.rows_eval == 0 ||
      task.rows_eval > task.rows || transform.shape.rows != task.rows ||
      transform.shape.bins != task.bins || transform.stride < task.rows * task.bins)
    throw std::invalid_argument("Ascend FFA detection task/transform shape mismatch");
  if (!(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp))
    throw std::invalid_argument("Ascend FFA detection effective_tsamp must be finite and > 0");
  if (!std::isfinite(search_options.snr_threshold))
    throw std::invalid_argument("Ascend FFA detection S/N threshold must be finite");
}

}  // namespace

AscendFfaBatchSearchResult detect_ffa_batch_ascend(
    AscendFfaBuffer transform, const FfaSearchTask& task,
    std::span<const std::size_t> width_trials,
    const FfaSearchOptions& search_options,
    const AscendLaunchOptions& launch_options) {
  validate_detection_input(transform, task, search_options, launch_options);
  const FfaDetectionPlan plan = make_ffa_detection_plan(width_trials, task.bins);
  const float stdnoise = ffa_task_stdnoise(task);
  const std::size_t trial_count = plan.boxcar_trials.size();
  const std::size_t record_count = checked_mul_detection(
      checked_mul_detection(transform.nseries, task.rows_eval,
                            "Ascend FFA detection record count overflow"),
      trial_count, "Ascend FFA detection record count overflow");

  std::vector<std::uint32_t> widths(trial_count);
  std::vector<float> heights(trial_count), baselines(trial_count);
  for (std::size_t i = 0; i < trial_count; ++i) {
    widths[i] = checked_u32(plan.boxcar_trials[i].width,
                            "Ascend FFA detection width exceeds uint32 range");
    heights[i] = plan.boxcar_trials[i].height;
    baselines[i] = plan.boxcar_trials[i].baseline;
  }

  check_acl_detection(aclrtSetDevice(launch_options.device_id),
                      "aclrtSetDevice for Ascend FFA detection");
  AscendDeviceBuffer<std::uint32_t> device_widths(trial_count, launch_options.device_id);
  AscendDeviceBuffer<float> device_heights(trial_count, launch_options.device_id);
  AscendDeviceBuffer<float> device_baselines(trial_count, launch_options.device_id);
  AscendDeviceBuffer<std::uint32_t> device_phases(record_count, launch_options.device_id);
  AscendDeviceBuffer<float> device_snrs(record_count, launch_options.device_id);
  check_acl_detection(aclrtMemcpy(device_widths.data(), device_widths.bytes(), widths.data(),
                                  widths.size() * sizeof(std::uint32_t), ACL_MEMCPY_HOST_TO_DEVICE),
                      "copy Ascend FFA detection widths");
  check_acl_detection(aclrtMemcpy(device_heights.data(), device_heights.bytes(), heights.data(),
                                  heights.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
                      "copy Ascend FFA detection heights");
  check_acl_detection(aclrtMemcpy(device_baselines.data(), device_baselines.bytes(), baselines.data(),
                                  baselines.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE),
                      "copy Ascend FFA detection baselines");

  const std::uint32_t block_dim = launch_options.block_dim == 0 ? 32U : launch_options.block_dim;
  gaffa_launch_ffa_detect_ascend(
      transform.data,
      checked_u32(transform.stride, "Ascend FFA transform stride exceeds uint32 range"),
      checked_u32(task.rows_eval, "Ascend FFA rows_eval exceeds uint32 range"),
      checked_u32(task.bins, "Ascend FFA bins exceeds uint32 range"),
      device_widths.data(), device_heights.data(), device_baselines.data(),
      checked_u32(trial_count, "Ascend FFA trial count exceeds uint32 range"),
      stdnoise, device_phases.data(), device_snrs.data(),
      checked_u32(transform.nseries, "Ascend FFA nseries exceeds uint32 range"),
      block_dim, launch_options.stream);
  check_acl_detection(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(launch_options.stream)),
                      "synchronize Ascend FFA detection");

  std::vector<std::uint32_t> phases(record_count);
  std::vector<float> snrs(record_count);
  check_acl_detection(aclrtMemcpy(phases.data(), phases.size() * sizeof(std::uint32_t),
                                  device_phases.data(), device_phases.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
                      "copy Ascend FFA detection phases");
  check_acl_detection(aclrtMemcpy(snrs.data(), snrs.size() * sizeof(float),
                                  device_snrs.data(), device_snrs.bytes(), ACL_MEMCPY_DEVICE_TO_HOST),
                      "copy Ascend FFA detection S/N");

  AscendFfaBatchSearchResult result;
  for (std::size_t series = 0; series < transform.nseries; ++series) {
    for (std::size_t shift = 0; shift < task.rows_eval; ++shift) {
      for (std::size_t trial = 0; trial < trial_count; ++trial) {
        const std::size_t index = (series * task.rows_eval + shift) * trial_count + trial;
        if (snrs[index] < search_options.snr_threshold) continue;
        if (search_options.max_peaks != 0 && result.peaks.size() >= search_options.max_peaks)
          throw std::runtime_error("Ascend FFA detection peak count exceeded max_peaks safety guard");
        result.peaks.push_back(AscendFfaBatchPeak{
            .series_index = series,
            .peak = make_ffa_peak(task, plan.boxcar_trials[trial], shift,
                                  phases[index], snrs[index])});
      }
    }
  }
  return result;
}

}  // namespace gaffa
