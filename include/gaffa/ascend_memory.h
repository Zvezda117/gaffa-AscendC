#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace gaffa {

template <typename T>
struct AscendSpan {
  T* data = nullptr;
  std::size_t count = 0;
  int device_id = 0;

  [[nodiscard]] std::size_t size() const noexcept { return count; }

  [[nodiscard]] std::size_t bytes() const noexcept {
    return count * sizeof(T);
  }

  [[nodiscard]] bool empty() const noexcept {
    return data == nullptr || count == 0;
  }
};

namespace detail {

template <typename T>
[[nodiscard]] constexpr std::size_t checked_ascend_buffer_bytes(
    std::size_t count) {
  if (count != 0 &&
      count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::overflow_error("Ascend device buffer size overflow");
  }
  return count * sizeof(T);
}

}  // namespace detail

class AscendDeviceMemory {
 public:
  AscendDeviceMemory() = default;
  explicit AscendDeviceMemory(std::size_t bytes, int device_id = 0);

  AscendDeviceMemory(const AscendDeviceMemory&) = delete;
  AscendDeviceMemory& operator=(const AscendDeviceMemory&) = delete;

  AscendDeviceMemory(AscendDeviceMemory&& other) noexcept;
  AscendDeviceMemory& operator=(AscendDeviceMemory&& other) noexcept;

  ~AscendDeviceMemory();

  void reset();

  [[nodiscard]] void* data() noexcept;
  [[nodiscard]] const void* data() const noexcept;
  [[nodiscard]] std::size_t bytes() const noexcept;
  [[nodiscard]] int device_id() const noexcept;

 private:
  void release_noexcept() noexcept;

  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  int device_id_ = -1;
};

template <typename T>
class AscendDeviceBuffer {
 public:
  AscendDeviceBuffer() = default;

  explicit AscendDeviceBuffer(std::size_t count, int device_id = 0)
      : memory_(detail::checked_ascend_buffer_bytes<T>(count), device_id),
        count_(count) {}

  [[nodiscard]] T* data() noexcept {
    return static_cast<T*>(memory_.data());
  }

  [[nodiscard]] const T* data() const noexcept {
    return static_cast<const T*>(memory_.data());
  }

  [[nodiscard]] T* get() noexcept { return data(); }
  [[nodiscard]] const T* get() const noexcept { return data(); }

  [[nodiscard]] AscendSpan<T> as_span() noexcept {
    return AscendSpan<T>{.data = data(), .count = count_, .device_id = memory_.device_id()};
  }

  [[nodiscard]] AscendSpan<const T> as_span() const noexcept {
    return AscendSpan<const T>{.data = data(), .count = count_, .device_id = memory_.device_id()};
  }

  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return memory_.bytes(); }
  [[nodiscard]] int device_id() const noexcept { return memory_.device_id(); }

 private:
  AscendDeviceMemory memory_{};
  std::size_t count_ = 0;
};

}  // namespace gaffa
