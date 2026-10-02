#pragma once
#include "backend_pool.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace flight::gateway {
class HealthMonitor {
 public:
  HealthMonitor(BackendPool& pool, std::chrono::milliseconds interval,
                std::chrono::milliseconds timeout, std::function<void(bool)> aggregate);
  ~HealthMonitor();
  void Start();
  void Stop();
 private:
  void Run();
  BackendPool& pool_;
  std::chrono::milliseconds interval_, timeout_;
  std::function<void(bool)> aggregate_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool stopping_{false};
  std::thread thread_;
};
}  // namespace flight::gateway
