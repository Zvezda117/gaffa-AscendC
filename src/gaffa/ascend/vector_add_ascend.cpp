#include "gaffa/vector_add_ascend.hpp"

#include "gaffa/ascend_memory.h"
#include "gaffa/ascend_runtime.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void gaffa_launch_vector_add_ascend(
    void* lhs, void* rhs, void* out, std::uint32_t total_elements,
    std::uint32_t block_dim, void* stream);

namespace gaffa {
namespace {

void check_acl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::size_t round_up(std::size_t value, std::size_t multiple) {
  if (value == 0) return 0;
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace

std::vector<float> vector_add_ascend(const std::vector<float>& lhs,
                                     const std::vector<float>& rhs,
                                     int device_id) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("vector_add_ascend requires equal input sizes");
  }
  if (lhs.empty()) return {};

  AscendRuntime runtime(device_id);
  constexpr std::size_t kTileElements = 256;
  constexpr std::uint32_t kMaxSmokeBlocks = 8;
  const std::uint32_t block_dim = static_cast<std::uint32_t>(
      std::min<std::size_t>(kMaxSmokeBlocks,
                            std::max<std::size_t>(1, (lhs.size() + kTileElements - 1) /
                                                       kTileElements)));
  const std::size_t alignment = kTileElements * block_dim;
  const std::size_t padded_count = round_up(lhs.size(), alignment);
  if (padded_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error("vector_add_ascend input exceeds uint32 kernel range");
  }

  std::vector<float> lhs_padded(padded_count, 0.0F);
  std::vector<float> rhs_padded(padded_count, 0.0F);
  std::copy(lhs.begin(), lhs.end(), lhs_padded.begin());
  std::copy(rhs.begin(), rhs.end(), rhs_padded.begin());

  AscendDeviceBuffer<float> device_lhs(padded_count, device_id);
  AscendDeviceBuffer<float> device_rhs(padded_count, device_id);
  AscendDeviceBuffer<float> device_out(padded_count, device_id);
  const std::size_t bytes = padded_count * sizeof(float);

  check_acl(aclrtMemcpy(device_lhs.data(), bytes, lhs_padded.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE), "copy lhs to Ascend");
  check_acl(aclrtMemcpy(device_rhs.data(), bytes, rhs_padded.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE), "copy rhs to Ascend");

  gaffa_launch_vector_add_ascend(device_lhs.data(), device_rhs.data(),
                                 device_out.data(),
                                 static_cast<std::uint32_t>(padded_count),
                                 block_dim, runtime.stream());
  runtime.synchronize();

  std::vector<float> out_padded(padded_count);
  check_acl(aclrtMemcpy(out_padded.data(), bytes, device_out.data(), bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "copy vector_add output from Ascend");
  out_padded.resize(lhs.size());
  return out_padded;
}

}  // namespace gaffa
