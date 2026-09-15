#include "gaffa/preprocessing_ascend.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gaffa {
namespace {
std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* message) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::overflow_error(message);
  return lhs + rhs;
}
std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error(message);
  return lhs * rhs;
}
void validate_detrend_options(DetrendRunningMedianOptions options) {
  if (options.window_samples == 0 || options.window_samples % 2 == 0)
    throw std::invalid_argument(
        "Ascend running median window_samples must be odd and > 0");
  if (options.min_points == 0 || options.min_points % 2 == 0)
    throw std::invalid_argument(
        "Ascend running median min_points must be odd and > 0");
}
}  // namespace

AscendPreprocessLayout compile_ascend_preprocess_layout(
    const PreprocessPlan& plan,
    const AscendPreprocessExecutionOptions& options) {
  if (options.series_tile_size == 0)
    throw std::invalid_argument(
        "Ascend preprocessing series_tile_size must be > 0");
  if (options.max_nsamples == 0)
    throw std::invalid_argument(
        "Ascend preprocessing max_nsamples must be > 0");
  if (plan.steps.empty())
    throw std::invalid_argument(
        "Ascend preprocess plan must contain at least one step");

  AscendPreprocessLayout result;
  result.steps.reserve(plan.steps.size());
  std::size_t max_scrunched_per_series = 0;
  std::size_t max_baseline_per_series = 0;
  for (const PreprocessStep& step : plan.steps) {
    AscendPreprocessStepLayout layout{
        .kind = step.kind,
        .detrend_options = step.detrend_running_median,
        .normalise = step.normalise,
    };
    switch (step.kind) {
      case PreprocessStepKind::DetrendRunningMedian: {
        validate_detrend_options(step.detrend_running_median);
        const std::size_t factor = std::max<std::size_t>(
            1, step.detrend_running_median.window_samples /
                   step.detrend_running_median.min_points);
        const std::size_t median_window =
            factor == 1 ? step.detrend_running_median.window_samples
                        : step.detrend_running_median.min_points;
        if (median_window > 256)
          throw std::invalid_argument(
              "Ascend running median effective window must be <= 256");
        layout.scrunch_factor = factor;
        layout.median_window = median_window;
        if (factor == 1) {
          max_baseline_per_series =
              std::max(max_baseline_per_series, options.max_nsamples);
        } else {
          const std::size_t reduced = options.max_nsamples / factor;
          max_scrunched_per_series =
              std::max(max_scrunched_per_series, reduced);
          max_baseline_per_series =
              std::max(max_baseline_per_series, reduced);
        }
        break;
      }
      case PreprocessStepKind::Normalise:
        break;
    }
    result.steps.push_back(layout);
  }

  result.scrunched_count = checked_multiply(
      options.series_tile_size, max_scrunched_per_series,
      "Ascend preprocessing scrunched workspace size overflow");
  result.baseline_count = checked_multiply(
      options.series_tile_size, max_baseline_per_series,
      "Ascend preprocessing baseline workspace size overflow");
  result.partial_stats_count = 0;
  result.series_stats_count = 0;

  auto& shape = result.workspace;
  shape.series_tile_size = options.series_tile_size;
  shape.max_nsamples = options.max_nsamples;
  shape.scrunched_bytes = checked_multiply(
      result.scrunched_count, sizeof(float),
      "Ascend preprocessing scrunched byte size overflow");
  shape.baseline_bytes = checked_multiply(
      result.baseline_count, sizeof(float),
      "Ascend preprocessing baseline byte size overflow");
  shape.partial_stats_bytes = 0;
  shape.series_stats_bytes = 0;
  shape.status_bytes = checked_multiply(
      options.series_tile_size, sizeof(int),
      "Ascend preprocessing status byte size overflow");
  shape.total_bytes = checked_add(
      checked_add(shape.scrunched_bytes, shape.baseline_bytes,
                  "Ascend preprocessing workspace byte size overflow"),
      checked_add(
          checked_add(shape.partial_stats_bytes, shape.series_stats_bytes,
                      "Ascend preprocessing workspace byte size overflow"),
          shape.status_bytes,
          "Ascend preprocessing workspace byte size overflow"),
      "Ascend preprocessing workspace byte size overflow");

  if (options.workspace_bytes_limit != 0 &&
      shape.total_bytes > options.workspace_bytes_limit)
    throw std::runtime_error(
        "Ascend preprocessing workspace exceeds workspace_bytes_limit");
  return result;
}

AscendPreprocessWorkspaceShape estimate_ascend_preprocess_workspace(
    const PreprocessPlan& plan,
    const AscendPreprocessExecutionOptions& options) {
  return compile_ascend_preprocess_layout(plan, options).workspace;
}

}  // namespace gaffa
