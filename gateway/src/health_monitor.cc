#include "health_monitor.h"

namespace flight::gateway {
HealthMonitor::HealthMonitor(BackendPool& pool, std::chrono::milliseconds interval,
                             std::chrono::milliseconds timeout,
                             std::function<void(bool)> aggregate)
    : pool_(pool), interval_(interval), timeout_(timeout), aggregate_(std::move(aggregate)) {}
HealthMonitor::~HealthMonitor() { Stop(); }
void HealthMonitor::Start() { thread_ = std::thread(&HealthMonitor::Run, this); }
void HealthMonitor::Stop() {
  { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
  changed_.notify_all();
  if (thread_.joinable()) thread_.join();
}
void HealthMonitor::Run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stopping_) {
    lock.unlock();
    const auto now = CircuitBreaker::Clock::now();
    pool_.ProbeHealth(timeout_, now);
    aggregate_(pool_.AggregateServing(now));
    lock.lock();
    changed_.wait_for(lock, interval_, [&] { return stopping_; });
  }
}
}  // namespace flight::gateway
