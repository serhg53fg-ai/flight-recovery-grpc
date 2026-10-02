#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace flight::gateway {

class NodeMetrics {
 public:
  struct Snapshot {
    std::uint64_t selected_count;
    std::uint64_t success_count;
    std::uint64_t failover_count;
    std::uint64_t total_latency_ms;
    std::map<std::string, std::uint64_t> failure_counts;
  };
  void Selected();
  void Success(std::uint64_t latency_ms);
  void Failure(const std::string& status, std::uint64_t latency_ms);
  void Failover();
  Snapshot Read() const;
 private:
  std::atomic<std::uint64_t> selected_{0}, success_{0}, failover_{0}, total_latency_ms_{0};
  mutable std::mutex failures_mutex_;
  std::map<std::string, std::uint64_t> failures_;
};

}  // namespace flight::gateway
