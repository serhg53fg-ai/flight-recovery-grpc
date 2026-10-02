#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flight::gateway {

struct WorkerConfig {
  std::string id;
  std::string address;
  std::uint32_t capacity;
  bool enabled;
};

struct GatewayConfig {
  std::string listen;
  std::uint32_t rpc_timeout_ms;
  std::uint32_t health_interval_ms;
  std::uint32_t health_timeout_ms;
  std::uint32_t failure_threshold;
  std::uint32_t open_cooldown_ms;
  std::uint32_t minimum_retry_budget_ms;
  std::vector<WorkerConfig> workers;
};

GatewayConfig LoadConfig(const std::string& path);

}  // namespace flight::gateway
