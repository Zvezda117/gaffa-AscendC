#pragma once

#include <cstdint>
#include <string>

namespace gaffa {

using AscendStream = void*;

class AscendRuntime {
 public:
  explicit AscendRuntime(int device_id = 0);
  AscendRuntime(const AscendRuntime&) = delete;
  AscendRuntime& operator=(const AscendRuntime&) = delete;
  AscendRuntime(AscendRuntime&& other) noexcept;
  AscendRuntime& operator=(AscendRuntime&& other) noexcept;
  ~AscendRuntime();

  [[nodiscard]] int device_id() const noexcept;
  [[nodiscard]] AscendStream stream() const noexcept;
  [[nodiscard]] std::string soc_name() const;

  void synchronize() const;

 private:
  void release_noexcept() noexcept;

  int device_id_ = -1;
  AscendStream stream_ = nullptr;
};

[[nodiscard]] int ascend_device_count();

}  // namespace gaffa
