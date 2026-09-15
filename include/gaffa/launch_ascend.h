#pragma once

#include "gaffa/ascend_runtime.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace gaffa {

enum class AscendGeneration { unknown, a2, a3 };

struct AscendDeviceProfile {
  int device_id = 0;
  AscendGeneration generation = AscendGeneration::unknown;
  std::string soc_name;
  std::size_t ai_core_count = 0;
  std::size_t ub_bytes = 0;

  [[nodiscard]] constexpr std::uint32_t npu_arch() const noexcept { return 2201; }
};

struct AscendLaunchOptions {
  int device_id = 0;
  std::uint32_t block_dim = 0;
  AscendStream stream = nullptr;
  bool synchronize_after_call = true;
};

}  // namespace gaffa
