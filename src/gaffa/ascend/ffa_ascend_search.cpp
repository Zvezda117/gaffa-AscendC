#include "gaffa/ffa_ascend.h"

#include <acl/acl.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace gaffa {
namespace {

void check_acl_search(aclError status, const char* operation) {
  if (status != ACL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
}

std::size_t checked_mul_search(std::size_t a, std::size_t b, const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error(message);
  return a * b;
}

void validate_search_input(AscendTimeSeriesBatchView input,
                           const FfaSearchPlan& plan,
                           const FfaSearchOptions& options,
                           const AscendLaunchOptions& launch) {
  if (input.data == nullptr || input.nseries == 0 || input.nsamples == 0)
    throw std::invalid_argument("Ascend FFA search input must be non-empty");
  if (input.device_id != launch.device_id || launch.device_id < 0)
    throw std::invalid_argument("Ascend FFA search input device must match launch device");
  if (!launch.synchronize_after_call)
    throw std::invalid_argument("Ascend FFA search requires synchronized launch");
  if (plan.tasks.empty()) throw std::invalid_argument("Ascend FFA search plan must contain tasks");
  if (plan.width_trials.empty()) throw std::invalid_argument("Ascend FFA search width trials must not be empty");
  if (!std::isfinite(options.snr_threshold)) throw std::invalid_argument("Ascend FFA search S/N threshold must be finite");
  for (const auto& task : plan.tasks) {
    if (task.input_nsamples != input.nsamples)
      throw std::invalid_argument("Ascend FFA search task input_nsamples must match batch nsamples");
    if (task.rows == 0 || task.bins <= 1 || task.rows_eval == 0 || task.rows_eval > task.rows)
      throw std::invalid_argument("Ascend FFA search task shape is invalid");
    const std::size_t elements = checked_mul_search(task.rows, task.bins,
                                                     "Ascend FFA search task size overflow");
    if (elements > task.prepared_nsamples)
      throw std::invalid_argument("Ascend FFA search task transform exceeds prepared samples");
  }
}

}  // namespace

AscendFfaBatchSearchResult search_ffa_batch_ascend(
    AscendTimeSeriesBatchView input, const FfaSearchPlan& plan,
    const FfaSearchOptions& search_options,
    const AscendLaunchOptions& launch_options) {
  validate_search_input(input, plan, search_options, launch_options);
  check_acl_search(aclrtSetDevice(launch_options.device_id),
                   "aclrtSetDevice for Ascend FFA search");
  AscendFfaBatchSearchResult result;
  for (const auto& task : plan.tasks) {
    const std::size_t prepared_count = checked_mul_search(input.nseries, task.prepared_nsamples,
                                                           "Ascend FFA prepared workspace size overflow");
    const std::size_t task_elements = checked_mul_search(task.rows, task.bins,
                                                          "Ascend FFA transform size overflow");
    const std::size_t transform_count = checked_mul_search(input.nseries, task_elements,
                                                            "Ascend FFA transform workspace size overflow");
    AscendDeviceBuffer<float> prepared(prepared_count, launch_options.device_id);
    AscendDeviceBuffer<float> scratch(transform_count, launch_options.device_id);
    AscendDeviceBuffer<float> output(transform_count, launch_options.device_id);
    prepare_ffa_input_ascend(input, task, prepared.as_span(), launch_options);
    const FfaTransformShape shape{.rows = task.rows, .bins = task.bins};
    ffa_transform_block_ascend(
        AscendFfaInput{.data = prepared.data(), .nseries = input.nseries,
                       .nsamples = task.prepared_nsamples, .stride = task.prepared_nsamples,
                       .shape = shape, .device_id = launch_options.device_id},
        AscendFfaBuffer{.data = scratch.data(), .nseries = input.nseries,
                        .stride = task_elements, .shape = shape,
                        .device_id = launch_options.device_id},
        AscendFfaBuffer{.data = output.data(), .nseries = input.nseries,
                        .stride = task_elements, .shape = shape,
                        .device_id = launch_options.device_id},
        launch_options);
    AscendFfaBatchSearchResult task_result = detect_ffa_batch_ascend(
        AscendFfaBuffer{.data = output.data(), .nseries = input.nseries,
                        .stride = task_elements, .shape = shape,
                        .device_id = launch_options.device_id},
        task, plan.width_trials, search_options, launch_options);
    result.peaks.insert(result.peaks.end(), task_result.peaks.begin(), task_result.peaks.end());
  }
  return result;
}

}  // namespace gaffa
