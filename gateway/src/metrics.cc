#include "metrics.h"

namespace flight::gateway {
void NodeMetrics::Selected() { selected_.fetch_add(1, std::memory_order_relaxed); }
void NodeMetrics::Success(std::uint64_t latency_ms) {
  success_.fetch_add(1, std::memory_order_relaxed);
  total_latency_ms_.fetch_add(latency_ms, std::memory_order_relaxed);
}
void NodeMetrics::Failure(const std::string& status, std::uint64_t latency_ms) {
  total_latency_ms_.fetch_add(latency_ms, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(failures_mutex_);
  ++failures_[status];
}
void NodeMetrics::Failover() { failover_.fetch_add(1, std::memory_order_relaxed); }
NodeMetrics::Snapshot NodeMetrics::Read() const {
  std::lock_guard<std::mutex> lock(failures_mutex_);
  return {selected_.load(), success_.load(), failover_.load(), total_latency_ms_.load(), failures_};
}
}  // namespace flight::gateway
