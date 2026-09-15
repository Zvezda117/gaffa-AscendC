#pragma once

#include "gaffa/ascend_memory.h"
#include "gaffa/launch_ascend.h"
#include "gaffa/preprocessing.h"
#include "gaffa/time_series_ascend.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace gaffa {

struct AscendPreprocessProgramOptions { int device_id = 0; };

struct AscendPreprocessExecutionOptions {
  std::size_t series_tile_size = 16;
  std::size_t max_nsamples = 0;
  std::uint32_t block_dim = 0;
  std::size_t workspace_bytes_limit = 0;
  AscendStream stream = nullptr;
};

struct AscendPreprocessWorkspaceShape {
  std::size_t series_tile_size = 0;
  std::size_t max_nsamples = 0;
  std::size_t scrunched_bytes = 0;
  std::size_t baseline_bytes = 0;
  std::size_t partial_stats_bytes = 0;
  std::size_t series_stats_bytes = 0;
  std::size_t status_bytes = 0;
  std::size_t total_bytes = 0;
};

struct AscendPreprocessStepLayout {
  PreprocessStepKind kind = PreprocessStepKind::Normalise;
  DetrendRunningMedianOptions detrend_options{};
  NormaliseOptions normalise{};
  std::size_t scrunch_factor = 1;
  std::size_t median_window = 0;
};

struct AscendPreprocessLayout {
  std::vector<AscendPreprocessStepLayout> steps;
  AscendPreprocessWorkspaceShape workspace;
  std::size_t scrunched_count = 0;
  std::size_t baseline_count = 0;
  std::size_t partial_stats_count = 0;
  std::size_t series_stats_count = 0;
};

AscendPreprocessLayout compile_ascend_preprocess_layout(
    const PreprocessPlan& plan,
    const AscendPreprocessExecutionOptions& options = {});

AscendPreprocessWorkspaceShape estimate_ascend_preprocess_workspace(
    const PreprocessPlan& plan,
    const AscendPreprocessExecutionOptions& options = {});

struct AscendPreprocessProgramImpl;

class AscendPreprocessProgram {
 public:
  AscendPreprocessProgram(
      PreprocessPlan plan,
      const AscendPreprocessProgramOptions& program_options = {},
      const AscendPreprocessExecutionOptions& execution_options = {});
  ~AscendPreprocessProgram();
  AscendPreprocessProgram(AscendPreprocessProgram&&) noexcept;
  AscendPreprocessProgram& operator=(AscendPreprocessProgram&&) noexcept;
  AscendPreprocessProgram(const AscendPreprocessProgram&) = delete;
  AscendPreprocessProgram& operator=(const AscendPreprocessProgram&) = delete;

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] int device_id() const;
  [[nodiscard]] std::size_t tile_capacity() const;
  [[nodiscard]] std::size_t max_nsamples() const;
  [[nodiscard]] const AscendPreprocessWorkspaceShape& workspace_shape() const;
  void synchronize();

 private:
  friend void preprocess_time_series_batch_inplace_ascend(
      AscendPreprocessProgram& program, MutableAscendTimeSeriesBatchView batch);
  std::unique_ptr<AscendPreprocessProgramImpl> impl_;
};

void preprocess_time_series_batch_inplace_ascend(
    AscendPreprocessProgram& program, MutableAscendTimeSeriesBatchView batch);

}  // namespace gaffa
