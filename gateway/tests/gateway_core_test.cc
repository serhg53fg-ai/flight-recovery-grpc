#include "circuit_breaker.h"
#include "config.h"
#include "backend_pool.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << " CHECK failed: "            \
                << #condition << '\n';                                           \
      ++failures;                                                                \
    }                                                                            \
  } while (false)

std::filesystem::path WriteConfig(const std::string& body,
                                  const std::string& name) {
  auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream output(path);
  output << body;
  return path;
}

bool Rejects(const std::string& body, const std::string& name) {
  try {
    (void)flight::gateway::LoadConfig(WriteConfig(body, name).string());
    return false;
  } catch (const std::invalid_argument&) {
    return true;
  }
}

const char* ValidPrefix = R"({
  "listen":"127.0.0.1:50051",
  "rpc_timeout_ms":20000,
  "health_interval_ms":2000,
  "health_timeout_ms":500,
  "failure_threshold":3,
  "open_cooldown_ms":10000,
  "minimum_retry_budget_ms":500,
  "workers":)";

void TestConfig() {
  const auto valid = std::string(ValidPrefix) + R"([
    {"id":"a","address":"127.0.0.1:50052","capacity":1,"enabled":true},
    {"id":"b","address":"127.0.0.1:50053","capacity":4,"enabled":false}
  ]})";
  auto config = flight::gateway::LoadConfig(
      WriteConfig(valid, "flight-gateway-valid.json").string());
  CHECK(config.listen == "127.0.0.1:50051");
  CHECK(config.workers.size() == 2);
  CHECK(config.workers[1].capacity == 4);

  CHECK(Rejects(valid.substr(0, valid.size() - 1) + R"(,"typo":1})",
                "flight-gateway-unknown.json"));
  CHECK(Rejects(std::string(ValidPrefix) + R"([]})",
                "flight-gateway-empty.json"));
  CHECK(Rejects(std::string(ValidPrefix) + R"([
    {"id":"a","address":"x","capacity":1,"enabled":true},
    {"id":"a","address":"y","capacity":1,"enabled":true}]})",
                "flight-gateway-duplicate.json"));
  CHECK(Rejects(std::string(ValidPrefix) + R"([
    {"id":"a","address":"x","capacity":0,"enabled":true}]})",
                "flight-gateway-capacity.json"));
  CHECK(Rejects(std::string(ValidPrefix) + R"([
    {"id":"a","address":"x","capacity":1,"enabled":false}]})",
                "flight-gateway-disabled.json"));
  auto zero_duration = valid;
  zero_duration.replace(zero_duration.find("\"rpc_timeout_ms\":20000"), 22,
                        "\"rpc_timeout_ms\":0");
  CHECK(Rejects(zero_duration, "flight-gateway-zero-duration.json"));
  std::string too_many = std::string(ValidPrefix) + '[';
  for (int index = 0; index < 257; ++index) {
    if (index != 0) too_many += ',';
    too_many += "{\"id\":\"w" + std::to_string(index) +
                "\",\"address\":\"127.0.0.1:50052\",\"capacity\":1,"
                "\"enabled\":true}";
  }
  too_many += "]}";
  CHECK(Rejects(too_many, "flight-gateway-too-many.json"));
  auto bad_timeout = valid;
  bad_timeout.replace(bad_timeout.find("\"health_timeout_ms\":500"), 23,
                      "\"health_timeout_ms\":2000");
  CHECK(Rejects(bad_timeout, "flight-gateway-timeout.json"));
  for (const std::string invalid : {"10.attacker.example:443", "192.168.example:443",
                                    "172.16evil.example:443", "localhost:not-a-port",
                                    "127.0.0.1:+50052", "127.0.0.1: 50052"}) {
    auto bad_address = valid;
    const auto position = bad_address.find("127.0.0.1:50052");
    bad_address.replace(position, 15, invalid);
    CHECK(Rejects(bad_address, "flight-gateway-bad-address.json"));
  }
}

void TestCircuitBreaker() {
  using flight::gateway::CircuitBreaker;
  using namespace std::chrono_literals;
  CircuitBreaker breaker(3, 10s);
  const CircuitBreaker::TimePoint start{};
  for (int i = 0; i < 2; ++i) breaker.OnHealthFailure(start);
  CHECK(breaker.state(start) == CircuitBreaker::State::kClosed);
  breaker.OnHealthFailure(start);
  CHECK(breaker.state(start) == CircuitBreaker::State::kOpen);
  CHECK(!breaker.TryAcquire(start + 9s).has_value());
  auto probe = breaker.TryAcquire(start + 10s);
  CHECK(probe.has_value());
  CHECK(breaker.state(start + 10s) == CircuitBreaker::State::kHalfOpen);
  CHECK(!breaker.TryAcquire(start + 10s).has_value());
  probe->OnNeutralResult();
  auto next_probe = breaker.TryAcquire(start + 10s);
  CHECK(next_probe.has_value());
  next_probe->OnPredictionSuccess();
  CHECK(breaker.state(start + 10s) == CircuitBreaker::State::kClosed);

  breaker.OnUnavailable(start + 11s);
  breaker.OnUnavailable(start + 11s);
  breaker.OnUnavailable(start + 11s);
  auto retry_probe = breaker.TryAcquire(start + 21s);
  CHECK(retry_probe.has_value());
  retry_probe->OnUnavailable(start + 21s);
  CHECK(breaker.state(start + 21s) == CircuitBreaker::State::kOpen);
  breaker.OnHealthSuccess();
  CHECK(breaker.state(start + 21s) == CircuitBreaker::State::kOpen);

  CircuitBreaker consecutive(3, 10s);
  consecutive.OnHealthFailure(start);
  consecutive.OnHealthFailure(start);
  consecutive.OnHealthSuccess();
  consecutive.OnHealthFailure(start);
  consecutive.OnHealthFailure(start);
  CHECK(consecutive.state(start) == CircuitBreaker::State::kClosed);

}

void TestBackendPool() {
  using flight::gateway::AcquireKind;
  using flight::gateway::BackendPool;
  using flight::gateway::GatewayConfig;
  using flight::gateway::WorkerConfig;
  using namespace std::chrono_literals;
  GatewayConfig config{"127.0.0.1:50051", 20000, 2000, 500, 3, 10000, 500,
                       {WorkerConfig{"a", "127.0.0.1:50052", 1, true},
                        WorkerConfig{"b", "127.0.0.1:50053", 2, true}}};
  BackendPool pool(config, 7);
  pool.UpdateHealth("a", true, "");
  pool.UpdateHealth("b", true, "");
  const flight::gateway::CircuitBreaker::TimePoint now{};

  auto first = pool.Acquire({}, now);
  CHECK(first.kind == AcquireKind::kSelected);
  CHECK(first.permit != nullptr);
  const std::string selected = first.permit->node_id();
  auto excluded = pool.Acquire({selected}, now);
  CHECK(excluded.kind == AcquireKind::kSelected);
  CHECK(excluded.permit->node_id() != selected);
  excluded.permit.reset();
  first.permit.reset();

  auto a = pool.Acquire({"b"}, now);
  CHECK(a.kind == AcquireKind::kSelected);
  CHECK(a.permit->node_id() == "a");
  auto normalized = pool.Acquire({}, now);
  CHECK(normalized.kind == AcquireKind::kSelected);
  CHECK(normalized.permit->node_id() == "b");
  auto b_second = pool.Acquire({"a"}, now);
  CHECK(b_second.kind == AcquireKind::kSelected);
  auto full = pool.Acquire({}, now);
  CHECK(full.kind == AcquireKind::kAllEligibleFull);
  a.permit.reset();
  normalized.permit.reset();
  b_second.permit.reset();

  pool.UpdateHealth("a", false, "down");
  pool.UpdateHealth("b", false, "down");
  auto unavailable = pool.Acquire({}, now);
  CHECK(unavailable.kind == AcquireKind::kNoEligibleNode);

  pool.UpdateHealth("a", true, "");
  pool.UpdateHealth("b", true, "");
  std::set<std::string> selected_ids;
  for (int i = 0; i < 20; ++i) {
    auto result = pool.Acquire({}, now);
    CHECK(result.kind == AcquireKind::kSelected);
    selected_ids.insert(result.permit->node_id());
  }
  CHECK(selected_ids.size() == 2);

  for (std::uint32_t seed = 1; seed <= 100; ++seed) {
    BackendPool seeded(config, seed);
    seeded.UpdateHealth("a", true, "");
    seeded.UpdateHealth("b", true, "");
    std::set<std::string> ids;
    for (int request = 0; request < 6; ++request) {
      auto result = seeded.Acquire({}, now);
      ids.insert(result.permit->node_id());
    }
    CHECK(ids.size() == 2);
  }

  pool.UpdateHealth("a", true, "");
  pool.UpdateHealth("b", false, "down");
  for (int failure = 0; failure < 3; ++failure) {
    auto attempt = pool.Acquire({"b"}, now);
    attempt.permit->circuit_permit().OnUnavailable(now);
  }
  auto half_open_probe = pool.Acquire({"b"}, now + 10s);
  CHECK(half_open_probe.kind == AcquireKind::kSelected);
  CHECK(pool.AggregateServing(now + 10s));
}

}  // namespace

int main() {
  TestConfig();
  TestCircuitBreaker();
  TestBackendPool();
  return failures == 0 ? 0 : 1;
}
