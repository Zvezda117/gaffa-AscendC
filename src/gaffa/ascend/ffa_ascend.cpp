#include "gaffa/ffa_ascend.h"

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
#include <vector>

extern "C" void gaffa_launch_ffa_copy_level_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, void*);
extern "C" void gaffa_launch_ffa_merge_level_ascend(
    void*, void*, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, void*, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, void*);

namespace gaffa {
namespace {

void check_acl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::uint32_t checked_u32(std::size_t value, const char* message) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::overflow_error(message);
  return static_cast<std::uint32_t>(value);
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::overflow_error(message);
  return lhs * rhs;
}

std::uint32_t role_value(AscendFfaBufferRole role) { return static_cast<std::uint32_t>(role); }

std::vector<std::uint32_t> pack_copy_ops(std::span<const AscendFfaCopyOp> ops) {
  std::vector<std::uint32_t> packed;
  packed.reserve(checked_multiply(ops.size(), std::size_t{5}, "Ascend FFA copy metadata overflow"));
  for (const auto& op : ops) {
    packed.push_back(role_value(op.input_role));
    packed.push_back(role_value(op.output_role));
    packed.push_back(op.input_begin_row);
    packed.push_back(op.output_begin_row);
    packed.push_back(op.rows);
  }
  return packed;
}

std::vector<std::uint32_t> pack_merge_ops(std::span<const AscendFfaMergeOp> ops) {
  std::vector<std::uint32_t> packed;
  packed.reserve(checked_multiply(ops.size(), std::size_t{9}, "Ascend FFA merge metadata overflow"));
  for (const auto& op : ops) {
    packed.push_back(role_value(op.head_role)); packed.push_back(role_value(op.tail_role));
    packed.push_back(role_value(op.output_role)); packed.push_back(op.head_begin_row);
    packed.push_back(op.tail_begin_row); packed.push_back(op.output_begin_row);
    packed.push_back(op.head_rows); packed.push_back(op.tail_rows);
    packed.push_back(op.output_rows);
  }
  return packed;
}

std::uint32_t max_copy_rows(std::span<const AscendFfaCopyOp> ops) {
  std::uint32_t value = 0; for (const auto& op : ops) value = std::max(value, op.rows); return value;
}
std::uint32_t max_merge_rows(std::span<const AscendFfaMergeOp> ops) {
  std::uint32_t value = 0; for (const auto& op : ops) value = std::max(value, op.output_rows); return value;
}
std::uint32_t choose_block_dim(std::size_t work_items, const AscendLaunchOptions& options) {
  if (options.block_dim != 0) return options.block_dim;
  constexpr std::size_t kConservativeCommonCoreLimit = 24;
  return static_cast<std::uint32_t>(std::max<std::size_t>(1, std::min(work_items, kConservativeCommonCoreLimit)));
}

class StreamLease {
 public:
  explicit StreamLease(const AscendLaunchOptions& options) {
    if (options.device_id < 0) throw std::invalid_argument("Ascend FFA device_id must be >= 0");
    if (!options.synchronize_after_call) throw std::invalid_argument("materialized Ascend FFA transform requires synchronized launch");
    if (options.stream == nullptr) { runtime_.emplace(options.device_id); stream_ = runtime_->stream(); }
    else { check_acl(aclrtSetDevice(options.device_id), "aclrtSetDevice for Ascend FFA stream"); stream_ = options.stream; }
  }
  [[nodiscard]] void* stream() const noexcept { return stream_; }
  void synchronize() const {
    if (runtime_.has_value()) runtime_->synchronize();
    else check_acl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_)), "aclrtSynchronizeStream for Ascend FFA");
  }
 private:
  std::optional<AscendRuntime> runtime_; void* stream_ = nullptr;
};

void validate_transform_arguments(AscendFfaInput input, AscendFfaBuffer scratch,
                                  AscendFfaBuffer output,
                                  const AscendLaunchOptions& options) {
  if (input.data == nullptr || scratch.data == nullptr || output.data == nullptr)
    throw std::invalid_argument("Ascend FFA buffers must not be null");
  if (input.nseries == 0 || scratch.nseries != input.nseries || output.nseries != input.nseries)
    throw std::invalid_argument("Ascend FFA buffer nseries values must match and be > 0");
  if (input.shape.rows != scratch.shape.rows || input.shape.bins != scratch.shape.bins ||
      input.shape.rows != output.shape.rows || input.shape.bins != output.shape.bins)
    throw std::invalid_argument("Ascend FFA buffer shapes must match");
  const std::size_t elements = checked_multiply(input.shape.rows, input.shape.bins,
                                                  "Ascend FFA transform element count overflow");
  if (input.nsamples < elements || input.stride < elements || scratch.stride < elements || output.stride < elements)
    throw std::invalid_argument("Ascend FFA buffer strides must contain the complete transform block");
  if (input.device_id != options.device_id || scratch.device_id != options.device_id || output.device_id != options.device_id)
    throw std::invalid_argument("Ascend FFA buffer device ids must match launch device_id");
}

void copy_metadata(AscendDeviceBuffer<std::uint32_t>& dst,
                   std::span<const std::uint32_t> src, const char* operation) {
  if (src.empty()) return;
  check_acl(aclrtMemcpy(dst.data(), dst.bytes(), src.data(), src.size_bytes(), ACL_MEMCPY_HOST_TO_DEVICE), operation);
}

}  // namespace

void ffa_transform_block_ascend(
    AscendFfaInput input, AscendFfaBuffer scratch, AscendFfaBuffer output,
    const AscendLaunchOptions& options) {
  validate_transform_arguments(input, scratch, output, options);
  const auto schedule = compile_ascend_ffa_transform_schedule(input.shape);
  const auto input_stride = checked_u32(input.stride, "Ascend FFA input stride exceeds uint32 range");
  const auto scratch_stride = checked_u32(scratch.stride, "Ascend FFA scratch stride exceeds uint32 range");
  const auto output_stride = checked_u32(output.stride, "Ascend FFA output stride exceeds uint32 range");
  const auto bins = checked_u32(input.shape.bins, "Ascend FFA bins exceed uint32 range");
  const auto nseries = checked_u32(input.nseries, "Ascend FFA nseries exceeds uint32 range");
  StreamLease stream(options);
  for (const auto& level : schedule.levels) {
    if (!level.copy_ops.empty()) {
      const auto packed = pack_copy_ops(level.copy_ops);
      AscendDeviceBuffer<std::uint32_t> metadata(packed.size(), options.device_id);
      copy_metadata(metadata, packed, "copy Ascend FFA copy metadata");
      const std::size_t work_items = checked_multiply(
          checked_multiply(input.nseries, level.copy_ops.size(), "Ascend FFA copy work item overflow"),
          max_copy_rows(level.copy_ops), "Ascend FFA copy work item overflow");
      gaffa_launch_ffa_copy_level_ascend(
          const_cast<float*>(input.data), scratch.data, output.data,
          input_stride, scratch_stride, output_stride, bins, metadata.data(),
          checked_u32(level.copy_ops.size(), "Ascend FFA copy op count exceeds uint32 range"),
          nseries, max_copy_rows(level.copy_ops), choose_block_dim(work_items, options), stream.stream());
      stream.synchronize();
    }
    if (!level.merge_ops.empty()) {
      const auto packed = pack_merge_ops(level.merge_ops);
      AscendDeviceBuffer<std::uint32_t> metadata(packed.size(), options.device_id);
      copy_metadata(metadata, packed, "copy Ascend FFA merge metadata");
      const std::size_t work_items = checked_multiply(
          checked_multiply(input.nseries, level.merge_ops.size(), "Ascend FFA merge work item overflow"),
          max_merge_rows(level.merge_ops), "Ascend FFA merge work item overflow");
      gaffa_launch_ffa_merge_level_ascend(
          const_cast<float*>(input.data), scratch.data, output.data,
          input_stride, scratch_stride, output_stride, bins, metadata.data(),
          checked_u32(level.merge_ops.size(), "Ascend FFA merge op count exceeds uint32 range"),
          nseries, max_merge_rows(level.merge_ops), choose_block_dim(work_items, options), stream.stream());
      stream.synchronize();
    }
  }
}

}  // namespace gaffa
