#include "gaffa/time_series_ascend.h"

#include "gaffa/ascend_runtime.h"
#include "gaffa/time_series.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

extern "C" void gaffa_launch_downsample_weighted_sum_ascend(
    void* input, void* first_input, void* last_input, void* first_weight,
    void* last_weight, void* output, std::uint32_t nseries,
    std::uint32_t input_nsamples, std::uint32_t output_nsamples,
    std::uint32_t block_dim, void* stream);
extern "C" void gaffa_launch_convert_uint32_to_float_ascend(
    void* input, void* output, std::uint32_t count, std::uint32_t block_dim,
    void* stream);

namespace gaffa {
namespace {
void check_acl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
  }
}
std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error(message);
  return lhs * rhs;
}
std::uint32_t checked_u32(std::size_t value, const char* message) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(message);
  return static_cast<std::uint32_t>(value);
}
std::uint32_t choose_block_dim(std::size_t work_items,
                               const AscendLaunchOptions& options) {
  if (options.block_dim != 0) return options.block_dim;
  constexpr std::size_t kConservativeCommonCoreLimit = 24;
  return static_cast<std::uint32_t>(std::max<std::size_t>(
      1, std::min(work_items, kConservativeCommonCoreLimit)));
}
struct StreamLease {
  explicit StreamLease(const AscendLaunchOptions& options)
      : external_stream(options.stream) {
    if (options.device_id < 0)
      throw std::invalid_argument("Ascend launch device_id must be >= 0");
    if (external_stream == nullptr) {
      owned_runtime.emplace(options.device_id);
      stream = owned_runtime->stream();
    } else {
      check_acl(aclrtSetDevice(options.device_id),
                "aclrtSetDevice for external Ascend stream");
      stream = external_stream;
    }
  }
  void synchronize_if_requested(const AscendLaunchOptions& options) const {
    if (!options.synchronize_after_call) return;
    if (owned_runtime.has_value()) owned_runtime->synchronize();
    else check_acl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)),
                   "aclrtSynchronizeStream");
  }
  AscendStream external_stream = nullptr;
  AscendStream stream = nullptr;
  mutable std::optional<AscendRuntime> owned_runtime;
};
void validate_downsample_arguments(AscendTimeSeriesBatchView input,
                                   double factor, AscendSpan<float> output,
                                   const AscendLaunchOptions& options,
                                   std::size_t output_nsamples) {
  if (input.data == nullptr) throw std::invalid_argument(
      "Ascend weighted downsample input data must not be null");
  if (input.nseries == 0) throw std::invalid_argument(
      "Ascend weighted downsample input nseries must be > 0");
  if (input.nsamples == 0) throw std::invalid_argument(
      "Ascend weighted downsample input nsamples must be > 0");
  if (output.data == nullptr) throw std::invalid_argument(
      "Ascend weighted downsample output data must not be null");
  if (input.device_id != options.device_id || output.device_id != options.device_id)
    throw std::invalid_argument(
        "Ascend weighted downsample device ids must match launch device_id");
  const std::size_t expected_output = checked_multiply(
      input.nseries, output_nsamples,
      "Ascend weighted downsample output element count overflow");
  if (output.count != expected_output) throw std::invalid_argument(
      "Ascend weighted downsample output size must match nseries * downsampled_size");
  (void)factor;
}
void validate_convert_arguments(AscendSpan<const std::uint32_t> input,
                                std::size_t nseries, std::size_t nsamples,
                                AscendSpan<float> output,
                                const AscendLaunchOptions& options) {
  if (options.device_id < 0)
    throw std::invalid_argument("Ascend launch device_id must be >= 0");
  if (input.data == nullptr || output.data == nullptr)
    throw std::invalid_argument(
        "Ascend time-series conversion input and output data must not be null");
  if (nseries == 0 || nsamples == 0)
    throw std::invalid_argument(
        "Ascend time-series conversion nseries and nsamples must be > 0");
  if (input.device_id != options.device_id || output.device_id != options.device_id)
    throw std::invalid_argument(
        "Ascend time-series conversion device ids must match launch device_id");
  const std::size_t count = checked_multiply(
      nseries, nsamples, "Ascend time-series conversion element count overflow");
  if (input.count != count || output.count != count)
    throw std::invalid_argument(
        "Ascend time-series conversion spans must match nseries * nsamples");
}
template <typename T>
void copy_metadata_to_device(AscendDeviceBuffer<T>& dst,
                             const std::vector<T>& src) {
  if (src.empty()) return;
  check_acl(aclrtMemcpy(dst.data(), dst.bytes(), src.data(), src.size() * sizeof(T),
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "copy Ascend time-series metadata");
}
}  // namespace

void downsample_weighted_sum_ascend(AscendTimeSeriesBatchView input,
                                    double factor, AscendSpan<float> output,
                                    const AscendLaunchOptions& options) {
  const auto plan = make_ascend_weighted_downsample_plan(input.nsamples, factor);
  validate_downsample_arguments(input, factor, output, options,
                                plan.output_nsamples);
  const auto nseries = checked_u32(
      input.nseries, "Ascend weighted downsample nseries exceeds uint32 range");
  const auto input_nsamples = checked_u32(
      input.nsamples, "Ascend weighted downsample input nsamples exceeds uint32 range");
  const auto output_nsamples = checked_u32(
      plan.output_nsamples, "Ascend weighted downsample output nsamples exceeds uint32 range");
  StreamLease stream_lease(options);
  AscendDeviceBuffer<std::uint32_t> first_input(plan.output_nsamples, options.device_id);
  AscendDeviceBuffer<std::uint32_t> last_input(plan.output_nsamples, options.device_id);
  AscendDeviceBuffer<float> first_weight(plan.output_nsamples, options.device_id);
  AscendDeviceBuffer<float> last_weight(plan.output_nsamples, options.device_id);
  copy_metadata_to_device(first_input, plan.first_input);
  copy_metadata_to_device(last_input, plan.last_input);
  copy_metadata_to_device(first_weight, plan.first_weight);
  copy_metadata_to_device(last_weight, plan.last_weight);
  const std::uint32_t block_dim = choose_block_dim(input.nseries, options);
  gaffa_launch_downsample_weighted_sum_ascend(
      const_cast<float*>(input.data), first_input.data(), last_input.data(),
      first_weight.data(), last_weight.data(), output.data, nseries,
      input_nsamples, output_nsamples, block_dim, stream_lease.stream);
  stream_lease.synchronize_if_requested(options);
}

void convert_time_series_batch_to_float_ascend(
    AscendSpan<const std::uint32_t> input, std::size_t nseries,
    std::size_t nsamples, AscendSpan<float> output,
    const AscendLaunchOptions& options) {
  validate_convert_arguments(input, nseries, nsamples, output, options);
  const std::size_t count = checked_multiply(
      nseries, nsamples, "Ascend time-series conversion element count overflow");
  const std::uint32_t count_u32 = checked_u32(
      count, "Ascend time-series conversion count exceeds uint32 range");
  StreamLease stream_lease(options);
  const std::uint32_t block_dim = choose_block_dim(count, options);
  gaffa_launch_convert_uint32_to_float_ascend(
      const_cast<std::uint32_t*>(input.data), output.data, count_u32,
      block_dim, stream_lease.stream);
  stream_lease.synchronize_if_requested(options);
}

}  // namespace gaffa
