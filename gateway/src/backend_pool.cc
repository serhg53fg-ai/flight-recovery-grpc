#include "backend_pool.h"
#include "gateway_logger.h"
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <algorithm>
#include <future>
#include <google/protobuf/struct.pb.h>
#include <utility>

namespace flight::gateway {
namespace {
void LogState(const std::string& event, const std::string& worker,
              const std::string& from, const std::string& to) {
  google::protobuf::Struct record;
  auto& fields = *record.mutable_fields();
  fields["event"].set_string_value(event);
  fields["worker_id"].set_string_value(worker);
  fields["from"].set_string_value(from);
  fields["to"].set_string_value(to);
  WriteGatewayLog(record);
}
const char* StateName(CircuitBreaker::State state) {
  if (state == CircuitBreaker::State::kClosed) return "CLOSED";
  if (state == CircuitBreaker::State::kOpen) return "OPEN";
  return "HALF_OPEN";
}
}  // namespace
struct BackendNode {
  BackendNode(const WorkerConfig& value, const GatewayConfig& config)
      : id(value.id), address(value.address), capacity(value.capacity), enabled(value.enabled),
        channel(grpc::CreateChannel(address, grpc::InsecureChannelCredentials())),
        prediction(flight::v1::InferenceWorker::NewStub(channel)),
        health(grpc::health::v1::Health::NewStub(channel)),
        breaker(config.failure_threshold, std::chrono::milliseconds(config.open_cooldown_ms)) {
    breaker.SetTransitionCallback([this](CircuitBreaker::State from,
                                         CircuitBreaker::State to) {
      LogState("gateway.circuit_transition", id, StateName(from), StateName(to));
    });
  }
  std::string id, address;
  std::uint32_t capacity;
  bool enabled;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<flight::v1::InferenceWorker::Stub> prediction;
  std::unique_ptr<grpc::health::v1::Health::Stub> health;
  CircuitBreaker breaker;
  std::atomic<std::uint32_t> inflight{0};
  std::atomic<bool> healthy{false};
  std::mutex status_mutex;
  std::string health_error;
  std::string prediction_error;
  std::int64_t last_health_unix_seconds{0};
  NodeMetrics metrics;
};

BackendPermit::BackendPermit(std::shared_ptr<BackendNode> node, CircuitBreaker::Permit permit)
    : node_(std::move(node)), circuit_permit_(std::move(permit)) {}
BackendPermit::~BackendPermit() { node_->inflight.fetch_sub(1, std::memory_order_acq_rel); }
const std::string& BackendPermit::node_id() const { return node_->id; }
flight::v1::InferenceWorker::Stub& BackendPermit::prediction_stub() { return *node_->prediction; }
CircuitBreaker::Permit& BackendPermit::circuit_permit() { return circuit_permit_; }
void BackendPermit::RecordSuccess(std::uint64_t value) {
  node_->metrics.Success(value);
  std::lock_guard<std::mutex> lock(node_->status_mutex);
  node_->prediction_error.clear();
}
void BackendPermit::RecordFailure(const std::string& status, const std::string& error,
                                  std::uint64_t value) {
  node_->metrics.Failure(status, value);
  std::lock_guard<std::mutex> lock(node_->status_mutex);
  node_->prediction_error = error.substr(0, 512);
}
void BackendPermit::RecordFailover() { node_->metrics.Failover(); }

BackendPool::BackendPool(const GatewayConfig& config, std::uint32_t seed)
    : random_(seed == 0 ? std::random_device{}() : seed) {
  for (const auto& worker : config.workers)
    nodes_.push_back(std::make_shared<BackendNode>(worker, config));
}

AcquireResult BackendPool::Acquire(const std::unordered_set<std::string>& excluded,
                                   CircuitBreaker::TimePoint now) {
  std::vector<std::shared_ptr<BackendNode>> routeable, available;
  for (const auto& node : nodes_) {
    if (!node->enabled || !node->healthy.load() || excluded.count(node->id) ||
        !node->breaker.CanAttempt(now)) continue;
    routeable.push_back(node);
    if (node->inflight.load() < node->capacity) available.push_back(node);
  }
  if (available.empty())
    return {routeable.empty() ? AcquireKind::kNoEligibleNode : AcquireKind::kAllEligibleFull,
            nullptr};
  const bool initially_full = routeable.size() > available.size();
  bool capacity_race = false;
  while (!available.empty()) {
    std::shared_ptr<BackendNode> chosen;
    if (available.size() == 1) chosen = available.front();
    else {
      std::size_t first, second;
      {
        std::lock_guard<std::mutex> lock(random_mutex_);
        std::uniform_int_distribution<std::size_t> pick(0, available.size() - 1);
        first = pick(random_);
        do { second = pick(random_); } while (second == first);
      }
      auto left = available[first], right = available[second];
      const auto left_load = std::uint64_t(left->inflight.load()) * right->capacity;
      const auto right_load = std::uint64_t(right->inflight.load()) * left->capacity;
      if (left_load == right_load) {
        auto lower = left->id < right->id ? left : right;
        auto upper = left->id < right->id ? right : left;
        chosen = tie_counter_.fetch_add(1) % 2 == 0 ? lower : upper;
      } else {
        chosen = left_load < right_load ? left : right;
      }
    }
    auto circuit = chosen->breaker.TryAcquire(now);
    auto current = chosen->inflight.load();
    while (circuit && current < chosen->capacity &&
           !chosen->inflight.compare_exchange_weak(current, current + 1)) {}
    if (circuit && current < chosen->capacity) {
      chosen->metrics.Selected();
      return {AcquireKind::kSelected,
              std::unique_ptr<BackendPermit>(new BackendPermit(chosen, std::move(*circuit)))};
    }
    if (circuit && current >= chosen->capacity) capacity_race = true;
    available.erase(std::remove(available.begin(), available.end(), chosen), available.end());
  }
  return {initially_full || capacity_race ? AcquireKind::kAllEligibleFull
                                         : AcquireKind::kNoEligibleNode,
          nullptr};
}

void BackendPool::UpdateHealth(const std::string& id, bool healthy, const std::string& error) {
  for (const auto& node : nodes_) if (node->id == id) {
    node->healthy.store(healthy);
    std::lock_guard<std::mutex> lock(node->status_mutex);
    node->health_error = error;
    return;
  }
}

void BackendPool::ProbeHealth(std::chrono::milliseconds timeout,
                              CircuitBreaker::TimePoint now) {
  std::vector<std::future<void>> probes;
  for (const auto& node : nodes_) {
    if (!node->enabled) continue;
    probes.push_back(std::async(std::launch::async, [node, timeout, now] {
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + timeout);
      grpc::health::v1::HealthCheckRequest request;
      request.set_service("flight.v1.InferenceWorker");
      grpc::health::v1::HealthCheckResponse response;
      const auto status = node->health->Check(&context, request, &response);
      const bool serving = status.ok() &&
          response.status() == grpc::health::v1::HealthCheckResponse::SERVING;
      const bool previous_health = node->healthy.exchange(serving);
      {
        std::lock_guard<std::mutex> lock(node->status_mutex);
        node->health_error = serving ? "" :
            (status.ok() ? "worker health is not SERVING" : status.error_message());
        node->last_health_unix_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
      }
      if (serving) node->breaker.OnHealthSuccess();
      else node->breaker.OnHealthFailure(now);
      if (previous_health != serving)
        LogState("gateway.health_transition", node->id,
                 previous_health ? "SERVING" : "NOT_SERVING",
                 serving ? "SERVING" : "NOT_SERVING");
    }));
  }
  for (auto& probe : probes) probe.get();
}

bool BackendPool::AggregateServing(CircuitBreaker::TimePoint now) {
  for (const auto& node : nodes_)
    if (node->enabled && node->healthy.load() &&
        node->breaker.state(now) != CircuitBreaker::State::kOpen) return true;
  return false;
}

std::vector<NodeSnapshot> BackendPool::Snapshot(CircuitBreaker::TimePoint now) {
  std::vector<NodeSnapshot> result;
  for (const auto& node : nodes_) {
    std::lock_guard<std::mutex> lock(node->status_mutex);
    result.push_back({node->id, node->address, node->enabled, node->healthy.load(),
                      node->breaker.state(now), node->inflight.load(), node->capacity,
                      node->prediction_error.empty() ? node->health_error
                                                     : node->prediction_error,
                      node->last_health_unix_seconds,
                      node->metrics.Read()});
  }
  return result;
}
}  // namespace flight::gateway
