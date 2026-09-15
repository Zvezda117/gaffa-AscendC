#include "gaffa/ascend_memory.h"
#include "gaffa/ascend_runtime.h"

#include <acl/acl.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace gaffa {
namespace {

[[noreturn]] void throw_acl(const char* operation, aclError status) {
  throw std::runtime_error(std::string(operation) + " failed with ACL error " +
                           std::to_string(static_cast<int>(status)));
}
void check_acl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) throw_acl(operation, status);
}

}  // namespace

AscendDeviceMemory::AscendDeviceMemory(std::size_t bytes, int device_id)
    : bytes_(bytes), device_id_(bytes == 0 ? -1 : device_id) {
  if (bytes_ == 0) return;
  detail::ensure_ascend_runtime_initialized();
  check_acl(aclrtSetDevice(device_id_), "aclrtSetDevice for aclrtMalloc");
  try {
    check_acl(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
  } catch (...) {
    bytes_ = 0;
    device_id_ = -1;
    throw;
  }
}

AscendDeviceMemory::AscendDeviceMemory(AscendDeviceMemory&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      device_id_(std::exchange(other.device_id_, -1)) {}

AscendDeviceMemory& AscendDeviceMemory::operator=(AscendDeviceMemory&& other) noexcept {
  if (this != &other) {
    release_noexcept();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    device_id_ = std::exchange(other.device_id_, -1);
  }
  return *this;
}

AscendDeviceMemory::~AscendDeviceMemory() { release_noexcept(); }

void AscendDeviceMemory::reset() {
  if (data_ == nullptr) return;
  int previous_device = -1;
  check_acl(aclrtGetDevice(&previous_device), "aclrtGetDevice");
  const bool switched = previous_device != device_id_;
  if (switched) check_acl(aclrtSetDevice(device_id_), "aclrtSetDevice for aclrtFree");
  const aclError free_status = aclrtFree(data_);
  const aclError restore_status = switched ? aclrtSetDevice(previous_device) : ACL_SUCCESS;
  if (free_status == ACL_SUCCESS) {
    data_ = nullptr;
    bytes_ = 0;
    device_id_ = -1;
  }
  if (restore_status != ACL_SUCCESS) {
    if (free_status != ACL_SUCCESS) {
      throw std::runtime_error("aclrtFree and device restoration both failed; free status=" +
                               std::to_string(static_cast<int>(free_status)) +
                               ", restore status=" +
                               std::to_string(static_cast<int>(restore_status)));
    }
    throw_acl("aclrtSetDevice restore after aclrtFree", restore_status);
  }
  check_acl(free_status, "aclrtFree");
}

void* AscendDeviceMemory::data() noexcept { return data_; }
const void* AscendDeviceMemory::data() const noexcept { return data_; }
std::size_t AscendDeviceMemory::bytes() const noexcept { return bytes_; }
int AscendDeviceMemory::device_id() const noexcept { return device_id_; }

void AscendDeviceMemory::release_noexcept() noexcept {
  if (data_ == nullptr) return;
  int previous_device = -1;
  if (aclrtGetDevice(&previous_device) != ACL_SUCCESS) return;
  const bool switched = previous_device != device_id_;
  if (switched && aclrtSetDevice(device_id_) != ACL_SUCCESS) return;
  if (aclrtFree(data_) == ACL_SUCCESS) {
    data_ = nullptr;
    bytes_ = 0;
    device_id_ = -1;
  }
  if (switched) (void)aclrtSetDevice(previous_device);
}

}  // namespace gaffa
