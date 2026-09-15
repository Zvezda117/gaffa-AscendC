#pragma once

#include "gaffa/ascend_memory.h"
#include "gaffa/dedispersion.h"
#include "gaffa/launch_ascend.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace gaffa {

struct AscendDedispersionOptions {
  int device_id = 0;
  std::uint32_t block_dim = 0;
  std::size_t time_tile_samples = 81920;
  AscendStream stream = nullptr;
};

struct AscendDedispersionLayout {
  std::vector<std::int32_t> delays;
  DedispersedShape output_shape{};
  std::size_t input_nsamples = 0;
  std::size_t input_nchans = 0;
  std::size_t channel_count = 0;
  std::size_t chan_begin = 0;
};

AscendDedispersionLayout compile_ascend_single_dm_layout(
    SampleShape shape, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan);
AscendDedispersionLayout compile_ascend_multi_dm_layout(
    SampleShape shape, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan);

struct AscendSubbandDedispersionLayout {
  std::vector<std::int32_t> coarse_delays;
  std::vector<std::int32_t> residual_delays;
  std::vector<std::uint32_t> subband_begin;
  std::vector<std::uint32_t> subband_end;
  DedispersedShape output_shape{};
  std::size_t input_nsamples = 0;
  std::size_t input_nchans = 0;
  std::size_t channel_count = 0;
  std::size_t chan_begin = 0;
  std::size_t subband_count = 0;
  std::size_t nominal_dm_count = 0;
  std::size_t ndm_per_nominal = 0;
  std::size_t max_residual_delay = 0;
};

AscendSubbandDedispersionLayout compile_ascend_subband_layout(
    SampleShape shape, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options = {});

template <typename T>
struct AscendDedispersedView {
  T* data = nullptr;
  DedispersedShape shape{};
  int device_id = 0;
  [[nodiscard]] AscendSpan<T> as_span() const noexcept {
    return AscendSpan<T>{.data = data, .count = size(), .device_id = device_id};
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return shape.ndm * shape.nsamples;
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return size() * sizeof(T); }
};

template <typename T>
struct AscendDedispersedResult {
  AscendDeviceBuffer<T> data;
  DedispersedShape shape{};
  int device_id = 0;
  [[nodiscard]] AscendDedispersedView<T> view() noexcept {
    return {.data = data.data(), .shape = shape, .device_id = device_id};
  }
  [[nodiscard]] AscendDedispersedView<const T> view() const noexcept {
    return {.data = data.data(), .shape = shape, .device_id = device_id};
  }
  [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
  [[nodiscard]] std::size_t bytes() const noexcept { return data.bytes(); }
};

template <typename T>
struct AscendDedispersedSpectrumView {
  T* data = nullptr;
  SampleShape shape{};
  int device_id = 0;
  [[nodiscard]] AscendSpan<T> as_span() const noexcept {
    return AscendSpan<T>{.data = data, .count = size(), .device_id = device_id};
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return shape.nsamples * shape.nchans;
  }
};

template <typename T>
struct AscendDedispersedSpectrum {
  AscendDeviceBuffer<T> data;
  SampleShape shape{};
  double dm = 0.0;
  double tsamp = 0.0;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;
  int device_id = 0;
  [[nodiscard]] AscendDedispersedSpectrumView<T> view() noexcept {
    return {.data = data.data(), .shape = shape, .device_id = device_id};
  }
  [[nodiscard]] AscendDedispersedSpectrumView<const T> view() const noexcept {
    return {.data = data.data(), .shape = shape, .device_id = device_id};
  }
  [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
  [[nodiscard]] std::size_t bytes() const noexcept { return data.bytes(); }
};

DedispersedResult<std::uint32_t> dedisperse_single_dm_ascend(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
DedispersedResult<std::uint32_t> dedisperse_single_dm_ascend(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
DedispersedResult<float> dedisperse_single_dm_ascend(
    HostSampleView<float>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});

DedispersedResult<std::uint32_t> dedisperse_multi_dm_ascend(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const AscendDedispersionOptions& = {});
DedispersedResult<std::uint32_t> dedisperse_multi_dm_ascend(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const AscendDedispersionOptions& = {});
DedispersedResult<float> dedisperse_multi_dm_ascend(
    HostSampleView<float>, std::span<const double>,
    const MultiDmDedispersionPlan&, const AscendDedispersionOptions& = {});

DedispersedResult<std::uint32_t> dedisperse_subband_ascend(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const SubbandDedispersionOptions& = {},
    const AscendDedispersionOptions& = {});
DedispersedResult<std::uint32_t> dedisperse_subband_ascend(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const SubbandDedispersionOptions& = {},
    const AscendDedispersionOptions& = {});
DedispersedResult<float> dedisperse_subband_ascend(
    HostSampleView<float>, std::span<const double>,
    const MultiDmDedispersionPlan&, const SubbandDedispersionOptions& = {},
    const AscendDedispersionOptions& = {});

DedispersedSpectrum<std::uint8_t> dedisperse_spectrum_ascend(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
DedispersedSpectrum<std::uint16_t> dedisperse_spectrum_ascend(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
DedispersedSpectrum<float> dedisperse_spectrum_ascend(
    HostSampleView<float>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});

AscendDedispersedResult<std::uint32_t> dedisperse_single_dm_ascend_device(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
AscendDedispersedResult<std::uint32_t> dedisperse_single_dm_ascend_device(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
AscendDedispersedResult<float> dedisperse_single_dm_ascend_device(
    HostSampleView<float>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});

AscendDedispersedResult<std::uint32_t> dedisperse_multi_dm_ascend_device(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const AscendDedispersionOptions& = {});
AscendDedispersedResult<std::uint32_t> dedisperse_multi_dm_ascend_device(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const AscendDedispersionOptions& = {});
AscendDedispersedResult<float> dedisperse_multi_dm_ascend_device(
    HostSampleView<float>, std::span<const double>,
    const MultiDmDedispersionPlan&, const AscendDedispersionOptions& = {});

AscendDedispersedResult<std::uint32_t> dedisperse_subband_ascend_device(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const SubbandDedispersionOptions& = {},
    const AscendDedispersionOptions& = {});
AscendDedispersedResult<std::uint32_t> dedisperse_subband_ascend_device(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const MultiDmDedispersionPlan&, const SubbandDedispersionOptions& = {},
    const AscendDedispersionOptions& = {});
AscendDedispersedResult<float> dedisperse_subband_ascend_device(
    HostSampleView<float>, std::span<const double>,
    const MultiDmDedispersionPlan&, const SubbandDedispersionOptions& = {},
    const AscendDedispersionOptions& = {});

AscendDedispersedSpectrum<std::uint8_t> dedisperse_spectrum_ascend_device(
    HostSampleView<std::uint8_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
AscendDedispersedSpectrum<std::uint16_t> dedisperse_spectrum_ascend_device(
    HostSampleView<std::uint16_t>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});
AscendDedispersedSpectrum<float> dedisperse_spectrum_ascend_device(
    HostSampleView<float>, std::span<const double>,
    const SingleDmDedispersionPlan&, const AscendDedispersionOptions& = {});

DedispersedResult<std::uint32_t> copy_to_host(
    const AscendDedispersedResult<std::uint32_t>&);
DedispersedResult<float> copy_to_host(
    const AscendDedispersedResult<float>&);
DedispersedSpectrum<std::uint8_t> copy_to_host(
    const AscendDedispersedSpectrum<std::uint8_t>&);
DedispersedSpectrum<std::uint16_t> copy_to_host(
    const AscendDedispersedSpectrum<std::uint16_t>&);
DedispersedSpectrum<float> copy_to_host(
    const AscendDedispersedSpectrum<float>&);

}  // namespace gaffa
