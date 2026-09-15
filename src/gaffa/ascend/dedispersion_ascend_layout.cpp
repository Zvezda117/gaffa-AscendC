#include "gaffa/dedispersion_ascend.h"

#include "gaffa/internal/dedispersion_delay.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace gaffa {
namespace {
void validate_shape_and_frequency(SampleShape shape,
                                  std::span<const double> frequency_mhz) {
  if (shape.nifs != 1)
    throw std::invalid_argument("Ascend dedispersion requires nifs == 1");
  if (shape.nsamples == 0 || shape.nchans == 0)
    throw std::invalid_argument(
        "Ascend dedispersion requires non-empty samples and channels");
  if (frequency_mhz.size() != shape.nchans)
    throw std::invalid_argument(
        "Ascend frequency table length must match channel count");
  for (double frequency : frequency_mhz) {
    if (!std::isfinite(frequency) || !(frequency > 0.0))
      throw std::invalid_argument(
          "Ascend frequency values must be finite and positive");
  }
}

void validate_common_plan(double ref_frequency_mhz, double tsamp,
                          std::size_t chan_begin, std::size_t chan_end,
                          std::size_t nchans,
                          std::span<const double> frequency_mhz) {
  if (!std::isfinite(ref_frequency_mhz) || !(ref_frequency_mhz > 0.0))
    throw std::invalid_argument(
        "Ascend reference frequency must be finite and positive");
  if (!std::isfinite(tsamp) || !(tsamp > 0.0))
    throw std::invalid_argument("Ascend tsamp must be finite and positive");
  if (chan_begin >= chan_end || chan_end > nchans)
    throw std::invalid_argument("Ascend dedispersion channel range is invalid");
  internal::validate_nonnegative_delay_range(
      frequency_mhz, ref_frequency_mhz, chan_begin, chan_end);
}
}  // namespace

AscendDedispersionLayout compile_ascend_single_dm_layout(
    SampleShape shape, std::span<const double> frequency_mhz,
    const SingleDmDedispersionPlan& plan) {
  validate_shape_and_frequency(shape, frequency_mhz);
  validate_common_plan(plan.ref_frequency_mhz, plan.tsamp, plan.chan_begin,
                       plan.chan_end, shape.nchans, frequency_mhz);
  if (!std::isfinite(plan.dm) || plan.dm < 0.0)
    throw std::invalid_argument("Ascend DM must be finite and non-negative");
  AscendDedispersionLayout layout;
  layout.delays = internal::make_single_dm_delay_table(
      frequency_mhz, plan.dm, plan.ref_frequency_mhz, plan.tsamp,
      plan.chan_begin, plan.chan_end);
  layout.output_shape = DedispersedShape{
      .ndm = 1,
      .nsamples = internal::valid_output_nsamples(
          shape.nsamples, internal::single_dm_max_delay(frequency_mhz, plan)),
  };
  layout.input_nsamples = shape.nsamples;
  layout.input_nchans = shape.nchans;
  layout.channel_count = plan.chan_end - plan.chan_begin;
  layout.chan_begin = plan.chan_begin;
  return layout;
}

AscendDedispersionLayout compile_ascend_multi_dm_layout(
    SampleShape shape, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan) {
  validate_shape_and_frequency(shape, frequency_mhz);
  validate_common_plan(plan.ref_frequency_mhz, plan.tsamp, plan.chan_begin,
                       plan.chan_end, shape.nchans, frequency_mhz);
  if (!std::isfinite(plan.dm_low) || plan.dm_low < 0.0)
    throw std::invalid_argument(
        "Ascend dm_low must be finite and non-negative");
  if (!std::isfinite(plan.dm_step) || !(plan.dm_step > 0.0))
    throw std::invalid_argument("Ascend dm_step must be finite and positive");
  if (plan.ndm == 0)
    throw std::invalid_argument("Ascend dedispersion requires ndm > 0");
  AscendDedispersionLayout layout;
  layout.delays = internal::make_multi_dm_delay_table(frequency_mhz, plan);
  layout.output_shape = DedispersedShape{
      .ndm = plan.ndm,
      .nsamples = internal::valid_output_nsamples(
          shape.nsamples, internal::multi_dm_max_delay(frequency_mhz, plan)),
  };
  layout.input_nsamples = shape.nsamples;
  layout.input_nchans = shape.nchans;
  layout.channel_count = plan.chan_end - plan.chan_begin;
  layout.chan_begin = plan.chan_begin;
  return layout;
}

AscendSubbandDedispersionLayout compile_ascend_subband_layout(
    SampleShape shape, std::span<const double> frequency_mhz,
    const MultiDmDedispersionPlan& plan,
    const SubbandDedispersionOptions& options) {
  const AscendDedispersionLayout direct =
      compile_ascend_multi_dm_layout(shape, frequency_mhz, plan);
  if (options.subband_channels == 0)
    throw std::invalid_argument("Ascend subband_channels must be positive");
  if (options.ndm_per_nominal == 0)
    throw std::invalid_argument("Ascend ndm_per_nominal must be positive");
  const auto ceil_div = [](std::size_t value, std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0);
  };
  AscendSubbandDedispersionLayout layout;
  layout.output_shape = direct.output_shape;
  layout.input_nsamples = direct.input_nsamples;
  layout.input_nchans = direct.input_nchans;
  layout.channel_count = direct.channel_count;
  layout.chan_begin = direct.chan_begin;
  layout.subband_count = ceil_div(layout.channel_count, options.subband_channels);
  layout.nominal_dm_count = ceil_div(plan.ndm, options.ndm_per_nominal);
  layout.ndm_per_nominal = options.ndm_per_nominal;
  layout.subband_begin.resize(layout.subband_count);
  layout.subband_end.resize(layout.subband_count);

  std::vector<double> subband_frequency(layout.subband_count);
  for (std::size_t subband = 0; subband < layout.subband_count; ++subband) {
    const std::size_t relative_begin = subband * options.subband_channels;
    const std::size_t relative_end =
        std::min(relative_begin + options.subband_channels, layout.channel_count);
    layout.subband_begin[subband] = static_cast<std::uint32_t>(relative_begin);
    layout.subband_end[subband] = static_cast<std::uint32_t>(relative_end);
    const std::size_t absolute_begin = plan.chan_begin + relative_begin;
    const std::size_t absolute_end = plan.chan_begin + relative_end;
    subband_frequency[subband] =
        frequency_mhz[(absolute_begin + absolute_end - 1) / 2];
  }

  layout.coarse_delays.resize(
      internal::checked_delay_table_size(layout.nominal_dm_count,
                                         layout.channel_count));
  for (std::size_t nominal = 0; nominal < layout.nominal_dm_count; ++nominal) {
    const double nominal_dm =
        plan.dm_low +
        static_cast<double>(nominal * options.ndm_per_nominal) * plan.dm_step;
    for (std::size_t c = 0; c < layout.channel_count; ++c) {
      layout.coarse_delays[nominal * layout.channel_count + c] =
          internal::dedispersion_delay_bins(
              nominal_dm, frequency_mhz[plan.chan_begin + c],
              plan.ref_frequency_mhz, plan.tsamp);
    }
  }

  layout.residual_delays.resize(
      internal::checked_delay_table_size(plan.ndm, layout.subband_count));
  std::int32_t max_residual = 0;
  for (std::size_t dm_index = 0; dm_index < plan.ndm; ++dm_index) {
    const std::size_t nominal = dm_index / options.ndm_per_nominal;
    const double nominal_dm =
        plan.dm_low +
        static_cast<double>(nominal * options.ndm_per_nominal) * plan.dm_step;
    const double final_dm =
        plan.dm_low + static_cast<double>(dm_index) * plan.dm_step;
    for (std::size_t subband = 0; subband < layout.subband_count; ++subband) {
      const std::int32_t delay = internal::dedispersion_delay_bins(
          final_dm - nominal_dm, subband_frequency[subband],
          plan.ref_frequency_mhz, plan.tsamp);
      if (delay < 0)
        throw std::logic_error("Ascend subband residual delay became negative");
      layout.residual_delays[dm_index * layout.subband_count + subband] = delay;
      max_residual = std::max(max_residual, delay);
    }
  }
  layout.max_residual_delay = static_cast<std::size_t>(max_residual);
  return layout;
}

}  // namespace gaffa
