#include "gaffa/dedispersion_ascend.h"

#include "gaffa/ascend_runtime.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

extern "C" void gaffa_launch_dedisperse_u8_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, void*);
extern "C" void gaffa_launch_dedisperse_u16_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, void*);
extern "C" void gaffa_launch_dedisperse_f32_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, void*);

extern "C" void gaffa_launch_dedisperse_spectrum_u8_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, void*);
extern "C" void gaffa_launch_dedisperse_spectrum_u16_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, void*);
extern "C" void gaffa_launch_dedisperse_spectrum_f32_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, void*);

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
      throw std::invalid_argument("Ascend dedispersion device_id must be >= 0");
    }
    if (options.stream == nullptr) {
      runtime_.emplace(options.device_id);
      stream_ = runtime_->stream();
    } else {
      check_acl(aclrtSetDevice(options.device_id),
                "aclrtSetDevice for Ascend dedispersion stream");
      stream_ = options.stream;
    }
  }

  [[nodiscard]] void* stream() const noexcept { return stream_; }

  void synchronize() const {
    if (runtime_.has_value()) {
      runtime_->synchronize();
    } else {
      check_acl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_)),
                "aclrtSynchronizeStream for Ascend dedispersion");
    }
  }

 private:
  std::optional<AscendRuntime> runtime_;
  void* stream_ = nullptr;
};

template <typename T>
void validate_host_samples(HostSampleView<T> samples) {
  if (samples.data.empty() || samples.data.data() == nullptr) {
    throw std::invalid_argument("Ascend dedispersion input samples must not be empty");
  }
  if (samples.data.size() != sample_element_count(samples.shape)) {
    throw std::invalid_argument("Ascend dedispersion sample data size does not match shape");
  }
}

template <typename T>
void copy_host_to_device(AscendDeviceBuffer<T>& dst, std::span<const T> src,
                         const char* operation) {
  if (src.empty()) return;
  check_acl(aclrtMemcpy(dst.data(), dst.bytes(), src.data(), src.size_bytes(),
                        ACL_MEMCPY_HOST_TO_DEVICE), operation);
}

template <typename T>
void copy_device_to_host(std::span<T> dst, const AscendDeviceBuffer<T>& src,
                         const char* operation) {
  if (dst.empty()) return;
  check_acl(aclrtMemcpy(dst.data(), dst.size_bytes(), src.data(), src.bytes(),
                        ACL_MEMCPY_DEVICE_TO_HOST), operation);
}

template <typename InT, typename OutT>
void launch_direct(void* input, void* delays, void* output, std::uint32_t ndm,
                   std::uint32_t input_nsamples, std::uint32_t input_nchans,
                   std::uint32_t channel_count, std::uint32_t chan_begin,
                   std::uint32_t output_nsamples, std::uint32_t block_dim,
                   void* stream) {
  if constexpr (std::is_same_v<InT, std::uint8_t>) {
    static_assert(std::is_same_v<OutT, std::uint32_t>);
    gaffa_launch_dedisperse_u8_ascend(input, delays, output, ndm,
                                      input_nsamples, input_nchans,
                                      channel_count, chan_begin,
                                      output_nsamples, block_dim, stream);
  } else if constexpr (std::is_same_v<InT, std::uint16_t>) {
    static_assert(std::is_same_v<OutT, std::uint32_t>);
    gaffa_launch_dedisperse_u16_ascend(input, delays, output, ndm,
                                       input_nsamples, input_nchans,
                                       channel_count, chan_begin,
                                       output_nsamples, block_dim, stream);
  } else {
    static_assert(std::is_same_v<InT, float> && std::is_same_v<OutT, float>);
    gaffa_launch_dedisperse_f32_ascend(input, delays, output, ndm,
                                       input_nsamples, input_nchans,
                                       channel_count, chan_begin,
                                       output_nsamples, block_dim, stream);
  }
}

template <typename T>
void launch_spectrum(void* input, void* delays, void* output,
                     std::uint32_t input_nsamples, std::uint32_t input_nchans,
                     std::uint32_t channel_count, std::uint32_t chan_begin,
                     std::uint32_t output_nsamples, std::uint32_t block_dim,
                     void* stream) {
  if constexpr (std::is_same_v<T, std::uint8_t>) {
    gaffa_launch_dedisperse_spectrum_u8_ascend(
        input, delays, output, input_nsamples, input_nchans, channel_count,
        chan_begin, output_nsamples, block_dim, stream);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    gaffa_launch_dedisperse_spectrum_u16_ascend(
        input, delays, output, input_nsamples, input_nchans, channel_count,
        chan_begin, output_nsamples, block_dim, stream);
  } else {
    static_assert(std::is_same_v<T, float>);
    gaffa_launch_dedisperse_spectrum_f32_ascend(
        input, delays, output, input_nsamples, input_nchans, channel_count,
        chan_begin, output_nsamples, block_dim, stream);
  }
}

template <typename InT, typename OutT, typename PlanT>
AscendDedispersedResult<OutT> dedisperse_direct_device_impl(
    HostSampleView<InT> samples, std::span<const double> frequency_mhz,
    const PlanT& plan, const AscendDedispersionOptions& options) {
  validate_host_samples(samples);
  const AscendDedispersionLayout layout = [&] {
    if constexpr (std::is_same_v<PlanT, SingleDmDedispersionPlan>) {
      return compile_ascend_single_dm_layout(samples.shape, frequency_mhz, plan);
    } else {
      return compile_ascend_multi_dm_layout(samples.shape, frequency_mhz, plan);
    }
  }();

  const std::size_t input_count = sample_element_count(samples.shape);
  const std::size_t output_count = dedispersed_element_count(layout.output_shape);
  StreamLease stream(options);
  AscendDeviceBuffer<InT> device_input(input_count, options.device_id);
  AscendDeviceBuffer<std::int32_t> device_delays(layout.delays.size(), options.device_id);
  AscendDedispersedResult<OutT> result{
      .data = AscendDeviceBuffer<OutT>(output_count, options.device_id),
      .shape = layout.output_shape,
      .device_id = options.device_id,
  };

  copy_host_to_device(device_input, samples.data, "copy Ascend dedispersion input");
  copy_host_to_device(device_delays, std::span<const std::int32_t>(layout.delays),
                      "copy Ascend dedispersion delays");

  const auto ndm = checked_u32(layout.output_shape.ndm, "Ascend ndm exceeds uint32 range");
  const auto input_nsamples = checked_u32(layout.input_nsamples, "Ascend input nsamples exceeds uint32 range");
  const auto input_nchans = checked_u32(layout.input_nchans, "Ascend input nchans exceeds uint32 range");
  const auto channel_count = checked_u32(layout.channel_count, "Ascend channel count exceeds uint32 range");
  const auto chan_begin = checked_u32(layout.chan_begin, "Ascend chan_begin exceeds uint32 range");
  const auto output_nsamples = checked_u32(layout.output_shape.nsamples, "Ascend output nsamples exceeds uint32 range");
  const auto block_dim = choose_block_dim(output_count, options);

  launch_direct<InT, OutT>(device_input.data(), device_delays.data(),
                            result.data.data(), ndm, input_nsamples,
                            input_nchans, channel_count, chan_begin,
                            output_nsamples, block_dim, stream.stream());
  stream.synchronize();
  return result;
}

template <typename T>
AscendDedispersedSpectrum<T> dedisperse_spectrum_device_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const AscendDedispersionOptions& options) {
  validate_host_samples(samples);
  const auto layout = compile_ascend_single_dm_layout(samples.shape, frequency_mhz, plan);
  const std::size_t output_count = checked_multiply(
      layout.output_shape.nsamples, layout.channel_count,
      "Ascend dedispersed spectrum output size overflow");
  StreamLease stream(options);
  AscendDeviceBuffer<T> device_input(samples.data.size(), options.device_id);
  AscendDeviceBuffer<std::int32_t> device_delays(layout.delays.size(), options.device_id);
  AscendDedispersedSpectrum<T> result{
      .data = AscendDeviceBuffer<T>(output_count, options.device_id),
      .shape = SampleShape{.nsamples = layout.output_shape.nsamples,
                           .nifs = 1,
                           .nchans = layout.channel_count},
      .dm = plan.dm,
      .tsamp = plan.tsamp,
      .chan_begin = plan.chan_begin,
      .chan_end = plan.chan_end,
      .device_id = options.device_id,
  };
  copy_host_to_device(device_input, samples.data, "copy Ascend spectrum input");
  copy_host_to_device(device_delays, std::span<const std::int32_t>(layout.delays),
                      "copy Ascend spectrum delays");
  launch_spectrum<T>(
      device_input.data(), device_delays.data(), result.data.data(),
      checked_u32(layout.input_nsamples, "Ascend spectrum input nsamples exceeds uint32 range"),
      checked_u32(layout.input_nchans, "Ascend spectrum input nchans exceeds uint32 range"),
      checked_u32(layout.channel_count, "Ascend spectrum channel count exceeds uint32 range"),
      checked_u32(layout.chan_begin, "Ascend spectrum chan_begin exceeds uint32 range"),
      checked_u32(layout.output_shape.nsamples, "Ascend spectrum output nsamples exceeds uint32 range"),
      choose_block_dim(output_count, options), stream.stream());
  stream.synchronize();
  return result;
}

template <typename InT, typename OutT, typename PlanT>
DedispersedResult<OutT> dedisperse_host_impl(
    HostSampleView<InT> samples, std::span<const double> frequency_mhz,
    const PlanT& plan, const AscendDedispersionOptions& options) {
  return copy_to_host(dedisperse_direct_device_impl<InT, OutT>(
      samples, frequency_mhz, plan, options));
}

template <typename T>
DedispersedSpectrum<T> dedisperse_spectrum_host_impl(
    HostSampleView<T> samples, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan,
    const AscendDedispersionOptions& options) {
  return copy_to_host(
      dedisperse_spectrum_device_impl(samples, frequency_mhz, plan, options));
}

}  // namespace

#define DEFINE_SINGLE(InT, OutT) \
AscendDedispersedResult<OutT> dedisperse_single_dm_ascend_device( \
    HostSampleView<InT> samples, std::span<const double> f, \
    const SingleDmDedispersionPlan& plan, const AscendDedispersionOptions& o) { \
  return dedisperse_direct_device_impl<InT, OutT>(samples, f, plan, o); \
} \
DedispersedResult<OutT> dedisperse_single_dm_ascend( \
    HostSampleView<InT> samples, std::span<const double> f, \
    const SingleDmDedispersionPlan& plan, const AscendDedispersionOptions& o) { \
  return dedisperse_host_impl<InT, OutT>(samples, f, plan, o); \
}

DEFINE_SINGLE(std::uint8_t, std::uint32_t)
DEFINE_SINGLE(std::uint16_t, std::uint32_t)
DEFINE_SINGLE(float, float)

#define DEFINE_MULTI(InT, OutT) \
AscendDedispersedResult<OutT> dedisperse_multi_dm_ascend_device( \
    HostSampleView<InT> samples, std::span<const double> f, \
    const MultiDmDedispersionPlan& plan, const AscendDedispersionOptions& o) { \
  return dedisperse_direct_device_impl<InT, OutT>(samples, f, plan, o); \
} \
DedispersedResult<OutT> dedisperse_multi_dm_ascend( \
    HostSampleView<InT> samples, std::span<const double> f, \
    const MultiDmDedispersionPlan& plan, const AscendDedispersionOptions& o) { \
  return dedisperse_host_impl<InT, OutT>(samples, f, plan, o); \
}

DEFINE_MULTI(std::uint8_t, std::uint32_t)
DEFINE_MULTI(std::uint16_t, std::uint32_t)
DEFINE_MULTI(float, float)

#define DEFINE_SPECTRUM(T) \
AscendDedispersedSpectrum<T> dedisperse_spectrum_ascend_device( \
    HostSampleView<T> samples, std::span<const double> f, \
    const SingleDmDedispersionPlan& plan, const AscendDedispersionOptions& o) { \
  return dedisperse_spectrum_device_impl(samples, f, plan, o); \
} \
DedispersedSpectrum<T> dedisperse_spectrum_ascend( \
    HostSampleView<T> samples, std::span<const double> f, \
    const SingleDmDedispersionPlan& plan, const AscendDedispersionOptions& o) { \
  return dedisperse_spectrum_host_impl(samples, f, plan, o); \
}

DEFINE_SPECTRUM(std::uint8_t)
DEFINE_SPECTRUM(std::uint16_t)
DEFINE_SPECTRUM(float)

DedispersedResult<std::uint32_t> copy_to_host(
    const AscendDedispersedResult<std::uint32_t>& result) {
  DedispersedResult<std::uint32_t> host;
  host.shape = result.shape;
  host.data.resize(result.size());
  copy_device_to_host(std::span<std::uint32_t>(host.data), result.data,
                      "copy Ascend uint32 dedispersion output to host");
  return host;
}

DedispersedResult<float> copy_to_host(
    const AscendDedispersedResult<float>& result) {
  DedispersedResult<float> host;
  host.shape = result.shape;
  host.data.resize(result.size());
  copy_device_to_host(std::span<float>(host.data), result.data,
                      "copy Ascend float dedispersion output to host");
  return host;
}

template <typename T>
DedispersedSpectrum<T> copy_spectrum_to_host_impl(
    const AscendDedispersedSpectrum<T>& result) {
  DedispersedSpectrum<T> host;
  host.shape = result.shape;
  host.dm = result.dm;
  host.tsamp = result.tsamp;
  host.chan_begin = result.chan_begin;
  host.chan_end = result.chan_end;
  host.data.resize(result.size());
  copy_device_to_host(std::span<T>(host.data), result.data,
                      "copy Ascend dedispersed spectrum to host");
  return host;
}

DedispersedSpectrum<std::uint8_t> copy_to_host(
    const AscendDedispersedSpectrum<std::uint8_t>& result) {
  return copy_spectrum_to_host_impl(result);
}
DedispersedSpectrum<std::uint16_t> copy_to_host(
    const AscendDedispersedSpectrum<std::uint16_t>& result) {
  return copy_spectrum_to_host_impl(result);
}
DedispersedSpectrum<float> copy_to_host(
    const AscendDedispersedSpectrum<float>& result) {
  return copy_spectrum_to_host_impl(result);
}

}  // namespace gaffa
