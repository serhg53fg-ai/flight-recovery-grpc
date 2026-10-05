#pragma once

#include "circuit_breaker.h"
#include "config.h"
#include "metrics.h"
#include <flight/v1/prediction.grpc.pb.h>
#include <grpc/health/v1/health.grpc.pb.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace flight::gateway {
struct BackendNode;

class BackendPermit {
 public:
  BackendPermit(const BackendPermit&) = delete;
  BackendPermit& operator=(const BackendPermit&) = delete;
  ~BackendPermit();
  const std::string& node_id() const;
  flight::v1::InferenceWorker::Stub& prediction_stub();
  CircuitBreaker::Permit& circuit_permit();
  void RecordSuccess(std::uint64_t latency_ms);
  void RecordFailure(const std::string& status, const std::string& error,
                     std::uint64_t latency_ms);
  void RecordFailover();
 private:
  friend class BackendPool;
  BackendPermit(std::shared_ptr<BackendNode> node, CircuitBreaker::Permit permit);
  std::shared_ptr<BackendNode> node_;
  CircuitBreaker::Permit circuit_permit_;
};

enum class AcquireKind { kSelected, kAllEligibleFull, kNoEligibleNode };
struct AcquireResult { AcquireKind kind; std::unique_ptr<BackendPermit> permit; };

struct NodeSnapshot {
  std::string id, address;
  bool enabled, healthy;
  CircuitBreaker::State circuit_state;
  std::uint32_t inflight, capacity;
  std::string last_error;
  std::int64_t last_health_unix_seconds;
  NodeMetrics::Snapshot metrics;
};

class BackendPool {
 public:
  explicit BackendPool(const GatewayConfig& config, std::uint32_t random_seed = 0);
  AcquireResult Acquire(const std::unordered_set<std::string>& excluded,
                        CircuitBreaker::TimePoint now);
  void UpdateHealth(const std::string& id, bool healthy, const std::string& error);
  void ProbeHealth(std::chrono::milliseconds timeout, CircuitBreaker::TimePoint now);
  bool AggregateServing(CircuitBreaker::TimePoint now);
  std::vector<NodeSnapshot> Snapshot(CircuitBreaker::TimePoint now);
 private:
  std::vector<std::shared_ptr<BackendNode>> nodes_;
  std::mt19937 random_;
  std::mutex random_mutex_;
  std::atomic<std::uint64_t> tie_counter_{0};
};
}  // namespace flight::gateway
