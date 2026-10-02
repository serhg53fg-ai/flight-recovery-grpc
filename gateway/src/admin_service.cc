#include "admin_service.h"
#include <chrono>

namespace flight::gateway {
grpc::Status AdminService::GetClusterStatus(grpc::ServerContext*, const google::protobuf::Empty*,
                                            flight::v1::ClusterStatus* response) {
  const auto now = CircuitBreaker::Clock::now();
  response->set_serving(pool_.AggregateServing(now));
  response->mutable_generated_at()->set_seconds(std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  for (const auto& snapshot : pool_.Snapshot(now)) {
    auto* node = response->add_nodes();
    node->set_id(snapshot.id); node->set_address(snapshot.address);
    node->set_enabled(snapshot.enabled); node->set_healthy(snapshot.healthy);
    node->set_inflight(snapshot.inflight); node->set_capacity(snapshot.capacity);
    node->set_last_error(snapshot.last_error);
    node->mutable_last_health_time()->set_seconds(snapshot.last_health_unix_seconds);
    node->set_circuit_state(snapshot.circuit_state == CircuitBreaker::State::kClosed
                                ? flight::v1::CLOSED
                                : snapshot.circuit_state == CircuitBreaker::State::kOpen
                                      ? flight::v1::OPEN : flight::v1::HALF_OPEN);
    node->set_selected_count(snapshot.metrics.selected_count);
    node->set_success_count(snapshot.metrics.success_count);
    node->set_failover_count(snapshot.metrics.failover_count);
    node->set_total_latency_ms(snapshot.metrics.total_latency_ms);
    for (const auto& [name, count] : snapshot.metrics.failure_counts)
      (*node->mutable_failure_counts())[name] = count;
  }
  return grpc::Status::OK;
}
}  // namespace flight::gateway
