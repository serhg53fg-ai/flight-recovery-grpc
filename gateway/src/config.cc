#include "config.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include <cmath>
#include <arpa/inet.h>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string_view>

namespace flight::gateway {
namespace {

using google::protobuf::Struct;
using google::protobuf::Value;

[[noreturn]] void Invalid(const std::string& message) {
  throw std::invalid_argument("invalid gateway config: " + message);
}

void ExactFields(const Struct& object, const std::set<std::string>& allowed,
                 const std::string& scope) {
  for (const auto& [key, unused] : object.fields()) {
    (void)unused;
    if (allowed.count(key) == 0) Invalid(scope + " has unknown field " + key);
  }
  for (const auto& key : allowed) {
    if (object.fields().count(key) == 0) Invalid(scope + " is missing " + key);
  }
}

const Value& Field(const Struct& object, const std::string& name) {
  const auto found = object.fields().find(name);
  if (found == object.fields().end()) Invalid("missing " + name);
  return found->second;
}

std::string String(const Struct& object, const std::string& name,
                   std::size_t maximum = 4096) {
  const auto& value = Field(object, name);
  if (value.kind_case() != Value::kStringValue || value.string_value().empty() ||
      value.string_value().size() > maximum) {
    Invalid(name + " must be a non-empty string");
  }
  return value.string_value();
}

std::uint32_t Integer(const Struct& object, const std::string& name,
                      std::uint32_t minimum, std::uint32_t maximum) {
  const auto& value = Field(object, name);
  const double number = value.number_value();
  if (value.kind_case() != Value::kNumberValue || !std::isfinite(number) ||
      std::floor(number) != number || number < minimum || number > maximum) {
    Invalid(name + " is outside its integer range");
  }
  return static_cast<std::uint32_t>(number);
}

bool Boolean(const Struct& object, const std::string& name) {
  const auto& value = Field(object, name);
  if (value.kind_case() != Value::kBoolValue) Invalid(name + " must be boolean");
  return value.bool_value();
}

bool IsTrustedAddress(std::string_view address) {
  const auto colon = address.rfind(':');
  if (colon == std::string_view::npos) return false;
  const auto host = address.substr(0, colon);
  const auto port_text = address.substr(colon + 1);
  if (port_text.empty() ||
      !std::all_of(port_text.begin(), port_text.end(),
                   [](char value) { return value >= '0' && value <= '9'; })) return false;
  try {
    std::size_t parsed = 0;
    const int port = std::stoi(std::string(port_text), &parsed);
    if (parsed != port_text.size() || port < 1 || port > 65535) return false;
  } catch (const std::exception&) {
    return false;
  }
  if (host == "localhost") return true;
  if (host == "[::1]") return true;
  in_addr parsed{};
  const std::string host_text(host);
  if (inet_pton(AF_INET, host_text.c_str(), &parsed) != 1) return false;
  const std::uint32_t ip = ntohl(parsed.s_addr);
  return (ip >> 24) == 127 || (ip >> 24) == 10 ||
         (ip >> 16) == ((192U << 8) | 168U) ||
         (ip >> 20) == ((172U << 4) | 1U);
}

}  // namespace

GatewayConfig LoadConfig(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open gateway config: " + path);
  const std::string json((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  Struct root;
  const auto status = google::protobuf::util::JsonStringToMessage(json, &root);
  if (!status.ok()) Invalid("JSON parse failed: " + std::string(status.message()));

  ExactFields(root,
              {"listen", "rpc_timeout_ms", "health_interval_ms",
               "health_timeout_ms", "failure_threshold", "open_cooldown_ms",
               "minimum_retry_budget_ms", "workers"},
              "root");
  GatewayConfig config{
      String(root, "listen"),
      Integer(root, "rpc_timeout_ms", 1, 600000),
      Integer(root, "health_interval_ms", 1, 60000),
      Integer(root, "health_timeout_ms", 1, 60000),
      Integer(root, "failure_threshold", 1, 256),
      Integer(root, "open_cooldown_ms", 1, 3600000),
      Integer(root, "minimum_retry_budget_ms", 1, 600000),
      {}};
  if (!IsTrustedAddress(config.listen)) Invalid("listen must use localhost or private IPv4");
  if (config.health_timeout_ms >= config.health_interval_ms) {
    Invalid("health_timeout_ms must be less than health_interval_ms");
  }

  const auto& workers = Field(root, "workers");
  if (workers.kind_case() != Value::kListValue || workers.list_value().values_size() < 1 ||
      workers.list_value().values_size() > 256) {
    Invalid("workers must contain 1..256 entries");
  }
  std::set<std::string> ids;
  bool any_enabled = false;
  for (int index = 0; index < workers.list_value().values_size(); ++index) {
    const auto& value = workers.list_value().values(index);
    if (value.kind_case() != Value::kStructValue) Invalid("worker must be an object");
    const auto& worker = value.struct_value();
    ExactFields(worker, {"id", "address", "capacity", "enabled"}, "worker");
    WorkerConfig parsed{String(worker, "id", 128), String(worker, "address"),
                        Integer(worker, "capacity", 1, 256),
                        Boolean(worker, "enabled")};
    if (!IsTrustedAddress(parsed.address)) {
      Invalid("worker address must use localhost or private IPv4");
    }
    if (!ids.insert(parsed.id).second) Invalid("worker id must be unique");
    any_enabled = any_enabled || parsed.enabled;
    config.workers.push_back(std::move(parsed));
  }
  if (!any_enabled) Invalid("at least one worker must be enabled");
  return config;
}

}  // namespace flight::gateway
