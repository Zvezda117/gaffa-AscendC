#include "gaffa/ascend_runtime.h"

#include <acl/acl.h>

#include <mutex>
#include <set>
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

class AclProcessRuntime {
 public:
  AclProcessRuntime() { check_acl(aclInit(nullptr), "aclInit"); }
  AclProcessRuntime(const AclProcessRuntime&) = delete;
  AclProcessRuntime& operator=(const AclProcessRuntime&) = delete;
  ~AclProcessRuntime() {
    std::set<int> devices;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      devices = devices_;
    }
    for (const int device : devices) (void)aclrtResetDevice(device);
    (void)aclFinalize();
  }
  void register_device(int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    devices_.insert(device_id);
  }
 private:
  std::mutex mutex_;
  std::set<int> devices_;
};

AclProcessRuntime& process_runtime() {
  static AclProcessRuntime runtime;
  return runtime;
}

}  // namespace

AscendRuntime::AscendRuntime(int device_id) : device_id_(device_id) {
  (void)process_runtime();
  check_acl(aclrtSetDevice(device_id_), "aclrtSetDevice");
  aclrtStream stream = nullptr;
  try {
    check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");
    stream_ = reinterpret_cast<AscendStream>(stream);
    process_runtime().register_device(device_id_);
  } catch (...) {
    device_id_ = -1;
    throw;
  }
}

AscendRuntime::AscendRuntime(AscendRuntime&& other) noexcept
    : device_id_(std::exchange(other.device_id_, -1)),
      stream_(std::exchange(other.stream_, nullptr)) {}

AscendRuntime& AscendRuntime::operator=(AscendRuntime&& other) noexcept {
  if (this != &other) {
    release_noexcept();
    device_id_ = std::exchange(other.device_id_, -1);
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

AscendRuntime::~AscendRuntime() { release_noexcept(); }
int AscendRuntime::device_id() const noexcept { return device_id_; }
AscendStream AscendRuntime::stream() const noexcept { return stream_; }

std::string AscendRuntime::soc_name() const {
  if (device_id_ < 0) return {};
  check_acl(aclrtSetDevice(device_id_), "aclrtSetDevice for aclrtGetSocName");
  const char* name = aclrtGetSocName();
  if (name == nullptr) throw std::runtime_error("aclrtGetSocName returned null");
  return name;
}

void AscendRuntime::synchronize() const {
  if (stream_ == nullptr) return;
  check_acl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_)),
            "aclrtSynchronizeStream");
}

void AscendRuntime::release_noexcept() noexcept {
  if (stream_ == nullptr) {
    device_id_ = -1;
    return;
  }
  if (device_id_ >= 0) (void)aclrtSetDevice(device_id_);
  (void)aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream_));
  stream_ = nullptr;
  device_id_ = -1;
}

int ascend_device_count() {
  (void)process_runtime();
  std::uint32_t count = 0;
  check_acl(aclrtGetDeviceCount(&count), "aclrtGetDeviceCount");
  return static_cast<int>(count);
}

}  // namespace gaffa
