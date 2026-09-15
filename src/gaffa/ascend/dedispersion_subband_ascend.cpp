#include "gaffa/dedispersion_ascend.h"

#include "gaffa/ascend_runtime.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

extern "C" void gaffa_launch_subband_stage1_u8_ascend(
    void*, void*, void*, void*, void*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, void*);
extern "C" void gaffa_launch_subband_stage1_u16_ascend(
    void*, void*, void*, void*, void*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, void*);
extern "C" void gaffa_launch_subband_stage1_f32_ascend(
    void*, void*, void*, void*, void*, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, void*);
extern "C" void gaffa_launch_subband_stage2_u32_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, void*);
extern "C" void gaffa_launch_subband_stage2_f32_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, void*);

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
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error(message);
  }
  return lhs + rhs;
}

std::uint32_t checked_u32(std::size_t value, const char* message) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error(message);
  }
  return static_cast<std::uint32_t>(value);
}

std::uint32_t choose_block_dim(std::size_t work_items,
                               const AscendDedispersionOptions& options) {
  if (options.block_dim != 0) return options.block_dim;
  constexpr std::size_t kConservativeCommonCoreLimit = 24;
  return static_cast<std::uint32_t>(
      std::max<std::size_t>(1, std::min(work_items, kConservativeCommonCoreLimit)));
}

class StreamLease {
 public:
  explicit StreamLease(const AscendDedispersionOptions& options) {
    if (options.device_id < 0) {
      throw std::invalid_argument("Ascend subband device_id must be >= 0");
    }
    if (options.stream == nullptr) {
      runtime_.emplace(options.device_id);
      stream_ = runtime_->stream();
    } else {
      check_acl(aclrtSetDevice(options.device_id),
                "aclrtSetDevice for Ascend subband stream");
      stream_ = options.stream;
    }
  }
  [[nodiscard]] void* stream() const noexcept { return stream_; }
  void synchronize() const {
    if (runtime_.has_value()) runtime_->synchronize();
    else check_acl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_)),
                   "aclrtSynchronizeStream for Ascend subband");
  }
 private:
  std::optional<AscendRuntime> runtime_;
  void* stream_ = nullptr;
};

template <typename T>
void copy_to_device(AscendDeviceBuffer<T>& dst, std::span<const T> src,
                    const char* operation) {
  if (src.empty()) return;
  check_acl(aclrtMemcpy(dst.data(), dst.bytes(), src.data(), src.size_bytes(),
                        ACL_MEMCPY_HOST_TO_DEVICE), operation);
}

template <typename InT, typename OutT>
void launch_stage1(void* input, void* coarse, void* sb_begin, void* sb_end,
                   void* intermediate, std::uint32_t nominal_dm_count,
                   std::uint32_t input_nsamples, std::uint32_t input_nchans,
                   std::uint32_t channel_count, std::uint32_t chan_begin,
                   std::uint32_t subband_count, std::uint32_t time_begin,
                   std::uint32_t intermediate_nsamples,
                   std::uint32_t block_dim, void* stream) {
  if constexpr (std::is_same_v<InT, std::uint8_t>) {
    static_assert(std::is_same_v<OutT, std::uint32_t>);
    gaffa_launch_subband_stage1_u8_ascend(
        input, coarse, sb_begin, sb_end, intermediate, nominal_dm_count,
        input_nsamples, input_nchans, channel_count, chan_begin, subband_count,
        time_begin, intermediate_nsamples, block_dim, stream);
  } else if constexpr (std::is_same_v<InT, std::uint16_t>) {
    static_assert(std::is_same_v<OutT, std::uint32_t>);
    gaffa_launch_subband_stage1_u16_ascend(
        input, coarse, sb_begin, sb_end, intermediate, nominal_dm_count,
        input_nsamples, input_nchans, channel_count, chan_begin, subband_count,
        time_begin, intermediate_nsamples, block_dim, stream);
  } else {
    static_assert(std::is_same_v<InT, float> && std::is_same_v<OutT, float>);
    gaffa_launch_subband_stage1_f32_ascend(
        input, coarse, sb_begin, sb_end, intermediate, nominal_dm_count,
        input_nsamples, input_nchans, channel_count, chan_begin, subband_count,
        time_begin, intermediate_nsamples, block_dim, stream);
  }
}

template <typename OutT>
void launch_stage2(void* intermediate, void* residual, void* output,
                   std::uint32_t ndm, std::uint32_t subband_count,
                   std::uint32_t ndm_per_nominal,
                   std::uint32_t intermediate_nsamples,
                   std::uint32_t output_tile_nsamples,
                   std::uint32_t output_total_nsamples,
                   std::uint32_t output_time_begin,
                   std::uint32_t block_dim, void* stream) {
  if constexpr (std::is_same_v<OutT, std::uint32_t>) {
    gaffa_launch_subband_stage2_u32_ascend(
        intermediate, residual, output, ndm, subband_count, ndm_per_nominal,
        intermediate_nsamples, output_tile_nsamples, output_total_nsamples,
        output_time_begin, block_dim, stream);
  } else {
    static_assert(std::is_same_v<OutT, float>);
    gaffa_launch_subband_stage2_f32_ascend(
        intermediate, residual, output, ndm, subband_count, ndm_per_nominal,
        intermediate_nsamples, output_tile_nsamples, output_total_nsamples,
        output_time_begin, block_dim, stream);
  }
}

template <typename InT, typename OutT>
AscendDedispersedResult<OutT> subband_device_impl(
    HostSampleView<InT> samples, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& subband_options,
    const AscendDedispersionOptions& options) {
  if (samples.data.empty() || samples.data.size() != sample_element_count(samples.shape)) {
    throw std::invalid_argument("Ascend subband sample data does not match shape");
  }
  if (options.time_tile_samples == 0) {
    throw std::invalid_argument("Ascend subband time_tile_samples must be > 0");
  }
  const auto layout = compile_ascend_subband_layout(
      samples.shape, frequency_mhz, plan, subband_options);
  const std::size_t output_count = dedispersed_element_count(layout.output_shape);
  const std::size_t max_output_tile =
      std::min(options.time_tile_samples, layout.output_shape.nsamples);
  const std::size_t max_intermediate_nsamples = checked_add(
      max_output_tile, layout.max_residual_delay,
      "Ascend subband intermediate time extent overflow");
  const std::size_t inter_series = checked_multiply(
      layout.nominal_dm_count, layout.subband_count,
      "Ascend subband intermediate series count overflow");
  const std::size_t intermediate_count = checked_multiply(
      inter_series, max_intermediate_nsamples,
      "Ascend subband intermediate element count overflow");

  StreamLease stream(options);
  AscendDeviceBuffer<InT> device_input(samples.data.size(), options.device_id);
  AscendDeviceBuffer<std::int32_t> device_coarse(layout.coarse_delays.size(), options.device_id);
  AscendDeviceBuffer<std::int32_t> device_residual(layout.residual_delays.size(), options.device_id);
  AscendDeviceBuffer<std::uint32_t> device_sb_begin(layout.subband_begin.size(), options.device_id);
  AscendDeviceBuffer<std::uint32_t> device_sb_end(layout.subband_end.size(), options.device_id);
  AscendDeviceBuffer<OutT> intermediate(intermediate_count, options.device_id);
  AscendDedispersedResult<OutT> result{
      .data = AscendDeviceBuffer<OutT>(output_count, options.device_id),
      .shape = layout.output_shape,
      .device_id = options.device_id,
  };

  copy_to_device(device_input, samples.data, "copy Ascend subband input");
  copy_to_device(device_coarse, std::span<const std::int32_t>(layout.coarse_delays),
                 "copy Ascend subband coarse delays");
  copy_to_device(device_residual, std::span<const std::int32_t>(layout.residual_delays),
                 "copy Ascend subband residual delays");
  copy_to_device(device_sb_begin, std::span<const std::uint32_t>(layout.subband_begin),
                 "copy Ascend subband starts");
  copy_to_device(device_sb_end, std::span<const std::uint32_t>(layout.subband_end),
                 "copy Ascend subband ends");

  const auto nominal_dm_count = checked_u32(layout.nominal_dm_count, "Ascend subband nominal DM count exceeds uint32 range");
  const auto input_nsamples = checked_u32(layout.input_nsamples, "Ascend subband input nsamples exceeds uint32 range");
  const auto input_nchans = checked_u32(layout.input_nchans, "Ascend subband input nchans exceeds uint32 range");
  const auto channel_count = checked_u32(layout.channel_count, "Ascend subband channel count exceeds uint32 range");
  const auto chan_begin = checked_u32(layout.chan_begin, "Ascend subband chan_begin exceeds uint32 range");
  const auto subband_count = checked_u32(layout.subband_count, "Ascend subband count exceeds uint32 range");
  const auto ndm = checked_u32(layout.output_shape.ndm, "Ascend subband ndm exceeds uint32 range");
  const auto ndm_per_nominal = checked_u32(layout.ndm_per_nominal, "Ascend ndm_per_nominal exceeds uint32 range");
  const auto output_total = checked_u32(layout.output_shape.nsamples, "Ascend subband output nsamples exceeds uint32 range");
  const auto stage1_blocks = choose_block_dim(inter_series, options);
  const auto stage2_blocks = choose_block_dim(layout.output_shape.ndm, options);

  for (std::size_t time_begin = 0; time_begin < layout.output_shape.nsamples;
       time_begin += options.time_tile_samples) {
    const std::size_t output_tile = std::min(
        options.time_tile_samples, layout.output_shape.nsamples - time_begin);
    const std::size_t intermediate_nsamples = checked_add(
        output_tile, layout.max_residual_delay,
        "Ascend subband tile halo overflow");
    launch_stage1<InT, OutT>(
        device_input.data(), device_coarse.data(), device_sb_begin.data(),
        device_sb_end.data(), intermediate.data(), nominal_dm_count,
        input_nsamples, input_nchans, channel_count, chan_begin, subband_count,
        checked_u32(time_begin, "Ascend subband time offset exceeds uint32 range"),
        checked_u32(intermediate_nsamples, "Ascend subband intermediate tile exceeds uint32 range"),
        stage1_blocks, stream.stream());
    launch_stage2<OutT>(
        intermediate.data(), device_residual.data(), result.data.data(), ndm,
        subband_count, ndm_per_nominal,
        checked_u32(intermediate_nsamples, "Ascend subband intermediate tile exceeds uint32 range"),
        checked_u32(output_tile, "Ascend subband output tile exceeds uint32 range"),
        output_total,
        checked_u32(time_begin, "Ascend subband output offset exceeds uint32 range"),
        stage2_blocks, stream.stream());
  }
  stream.synchronize();
  return result;
}

template <typename T>
DedispersedResult<T> copy_subband_to_host(const AscendDedispersedResult<T>& device) {
  DedispersedResult<T> result;
  result.shape = device.shape;
  result.data.resize(device.size());
  if (!result.data.empty()) {
    check_acl(aclrtMemcpy(result.data.data(), result.data.size() * sizeof(T),
                          device.data.data(), device.data.bytes(),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "copy Ascend subband output to host");
  }
  return result;
}

}  // namespace

#define DEFINE_SUBBAND(InT, OutT) \
AscendDedispersedResult<OutT> dedisperse_subband_ascend_device( \
    HostSampleView<InT> samples, std::span<const double> f, \
    const MultiDmDedispersionPlan& plan, \
    const SubbandDedispersionOptions& sb, const AscendDedispersionOptions& o) { \
  return subband_device_impl<InT, OutT>(samples, f, plan, sb, o); \
} \
DedispersedResult<OutT> dedisperse_subband_ascend( \
    HostSampleView<InT> samples, std::span<const double> f, \
    const MultiDmDedispersionPlan& plan, \
    const SubbandDedispersionOptions& sb, const AscendDedispersionOptions& o) { \
  return copy_subband_to_host( \
      subband_device_impl<InT, OutT>(samples, f, plan, sb, o)); \
}

DEFINE_SUBBAND(std::uint8_t, std::uint32_t)
DEFINE_SUBBAND(std::uint16_t, std::uint32_t)
DEFINE_SUBBAND(float, float)

}  // namespace gaffa
