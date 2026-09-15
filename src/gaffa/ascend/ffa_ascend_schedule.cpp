#include "gaffa/ffa_ascend.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gaffa {
namespace {

std::uint32_t checked_u32(std::size_t value, const char* message) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(message);
  return static_cast<std::uint32_t>(value);
}

void append_level(AscendFfaTransformLevel& output,
                  const AscendFfaTransformLevel& input) {
  output.copy_ops.insert(output.copy_ops.end(), input.copy_ops.begin(), input.copy_ops.end());
  output.merge_ops.insert(output.merge_ops.end(), input.merge_ops.begin(), input.merge_ops.end());
}

std::vector<AscendFfaTransformLevel> build_levels(
    AscendFfaBufferRole input_role, AscendFfaBufferRole scratch_role,
    AscendFfaBufferRole output_role, std::size_t row_begin, std::size_t rows) {
  if (rows == 1) {
    AscendFfaTransformLevel level;
    level.copy_ops.push_back(AscendFfaCopyOp{
        .input_role = input_role, .output_role = output_role,
        .input_begin_row = checked_u32(row_begin, "Ascend FFA input row exceeds uint32 range"),
        .output_begin_row = checked_u32(row_begin, "Ascend FFA output row exceeds uint32 range"),
        .rows = 1});
    return {level};
  }
  if (rows == 2) {
    AscendFfaTransformLevel level;
    level.merge_ops.push_back(AscendFfaMergeOp{
        .head_role = input_role, .tail_role = input_role,
        .output_role = output_role,
        .head_begin_row = checked_u32(row_begin, "Ascend FFA head row exceeds uint32 range"),
        .tail_begin_row = checked_u32(row_begin + 1, "Ascend FFA tail row exceeds uint32 range"),
        .output_begin_row = checked_u32(row_begin, "Ascend FFA output row exceeds uint32 range"),
        .head_rows = 1, .tail_rows = 1, .output_rows = 2});
    return {level};
  }
  const std::size_t head_rows = rows / 2;
  const std::size_t tail_rows = rows - head_rows;
  auto head_levels = build_levels(input_role, output_role, scratch_role, row_begin, head_rows);
  auto tail_levels = build_levels(input_role, output_role, scratch_role,
                                  row_begin + head_rows, tail_rows);
  std::vector<AscendFfaTransformLevel> levels(std::max(head_levels.size(), tail_levels.size()));
  for (std::size_t i = 0; i < head_levels.size(); ++i) append_level(levels[i], head_levels[i]);
  for (std::size_t i = 0; i < tail_levels.size(); ++i) append_level(levels[i], tail_levels[i]);
  AscendFfaTransformLevel root;
  root.merge_ops.push_back(AscendFfaMergeOp{
      .head_role = scratch_role, .tail_role = scratch_role,
      .output_role = output_role,
      .head_begin_row = checked_u32(row_begin, "Ascend FFA head row exceeds uint32 range"),
      .tail_begin_row = checked_u32(row_begin + head_rows, "Ascend FFA tail row exceeds uint32 range"),
      .output_begin_row = checked_u32(row_begin, "Ascend FFA output row exceeds uint32 range"),
      .head_rows = checked_u32(head_rows, "Ascend FFA head row count exceeds uint32 range"),
      .tail_rows = checked_u32(tail_rows, "Ascend FFA tail row count exceeds uint32 range"),
      .output_rows = checked_u32(rows, "Ascend FFA output row count exceeds uint32 range")});
  levels.push_back(std::move(root));
  return levels;
}

}  // namespace

AscendFfaTransformSchedule compile_ascend_ffa_transform_schedule(FfaTransformShape shape) {
  if (shape.rows == 0) throw std::invalid_argument("Ascend FFA transform rows must be > 0");
  if (shape.bins <= 1) throw std::invalid_argument("Ascend FFA transform bins must be > 1");
  checked_u32(shape.rows, "Ascend FFA transform rows exceed uint32 range");
  checked_u32(shape.bins, "Ascend FFA transform bins exceed uint32 range");
  if (shape.rows > std::numeric_limits<std::size_t>::max() / shape.bins)
    throw std::overflow_error("Ascend FFA transform element count overflow");
  return AscendFfaTransformSchedule{
      .shape = shape,
      .levels = build_levels(AscendFfaBufferRole::Input,
                             AscendFfaBufferRole::Scratch,
                             AscendFfaBufferRole::Output, 0, shape.rows)};
}

}  // namespace gaffa
