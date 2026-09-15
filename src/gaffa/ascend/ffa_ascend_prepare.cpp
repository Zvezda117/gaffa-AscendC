#include "gaffa/ffa_ascend.h"

#include "gaffa/time_series.h"

#include <acl/acl.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace gaffa {
namespace {

void check_acl_prepare(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::size_t checked_multiply_prepare(std::size_t lhs, std::size_t rhs,
                                     const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

}  // namespace

void prepare_ffa_input_ascend(
    AscendTimeSeriesBatchView input, const FfaSearchTask& task,
    AscendSpan<float> output, const AscendLaunchOptions& options) {
  if (options.device_id < 0) throw std::invalid_argument("Ascend FFA prepare device_id must be >= 0");
  if (!options.synchronize_after_call) throw std::invalid_argument("Ascend FFA prepare requires synchronized launch");
  if (input.data == nullptr || output.data == nullptr) throw std::invalid_argument("Ascend FFA prepare input and output must not be null");
  if (input.nseries == 0 || input.nsamples == 0) throw std::invalid_argument("Ascend FFA prepare input shape must be non-empty");
  if (input.device_id != options.device_id || output.device_id != options.device_id)
    throw std::invalid_argument("Ascend FFA prepare device ids must match launch device_id");
  if (task.input_nsamples != input.nsamples) throw std::invalid_argument("Ascend FFA prepare task input_nsamples must match input nsamples");
  if (!std::isfinite(task.downsample_factor) || task.downsample_factor < 1.0)
    throw std::invalid_argument("Ascend FFA prepare downsample_factor must be finite and >= 1");
  if (!(task.effective_tsamp > 0.0) || !std::isfinite(task.effective_tsamp))
    throw std::invalid_argument("Ascend FFA prepare effective_tsamp must be finite and > 0");
  std::size_t expected_prepared = input.nsamples;
  if (task.downsample_factor != 1.0) expected_prepared = downsampled_size(input.nsamples, task.downsample_factor);
  if (task.prepared_nsamples != expected_prepared)
    throw std::invalid_argument("Ascend FFA prepare prepared_nsamples does not match factor");
  const std::size_t task_elements = checked_multiply_prepare(task.rows, task.bins, "Ascend FFA task shape size overflow");
  if (task.bins <= 1 || task.rows == 0 || task.rows_eval == 0 ||
      task.rows_eval > task.rows || task_elements > task.prepared_nsamples)
    throw std::invalid_argument("Ascend FFA prepare task shape is invalid");
  const std::size_t expected_output = checked_multiply_prepare(
      input.nseries, task.prepared_nsamples,
      "Ascend FFA prepare output element count overflow");
  if (output.count != expected_output)
    throw std::invalid_argument("Ascend FFA prepare output size must match nseries * prepared_nsamples");

  check_acl_prepare(aclrtSetDevice(options.device_id),
                    "aclrtSetDevice for Ascend FFA prepare");
  if (task.downsample_factor == 1.0) {
    const std::size_t bytes = checked_multiply_prepare(expected_output, sizeof(float),
                                                        "Ascend FFA prepare byte size overflow");
    check_acl_prepare(aclrtMemcpy(output.data, bytes, input.data, bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE),
                      "copy Ascend FFA prepared input");
    return;
  }
  downsample_weighted_sum_ascend(input, task.downsample_factor, output, options);
}

}  // namespace gaffa
