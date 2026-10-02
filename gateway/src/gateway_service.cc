#include "gateway_service.h"
#include "gateway_logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <regex>
#include <unordered_set>
#include <google/protobuf/struct.pb.h>

namespace flight::gateway {
namespace {
using Timestamp = google::protobuf::Timestamp;
void LogAttempt(const std::string& trace, int attempt, const std::string& worker,
                bool failover, int status, std::int64_t latency_ms) {
  google::protobuf::Struct record;
  auto& fields = *record.mutable_fields();
  fields["event"].set_string_value("gateway.attempt");
  fields["trace_id"].set_string_value(trace.substr(0, 128));
  fields["attempt"].set_number_value(attempt);
  fields["worker_id"].set_string_value(worker);
  fields["failover"].set_bool_value(failover);
  fields["status"].set_number_value(status);
  fields["latency_ms"].set_number_value(latency_ms);
  WriteGatewayLog(record);
}
std::string StatusLabel(const grpc::Status& status) {
  const char* name = "UNKNOWN";
  switch (status.error_code()) {
    case grpc::StatusCode::INVALID_ARGUMENT: name = "INVALID_ARGUMENT"; break;
    case grpc::StatusCode::DEADLINE_EXCEEDED: name = "DEADLINE_EXCEEDED"; break;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: name = "RESOURCE_EXHAUSTED"; break;
    case grpc::StatusCode::CANCELLED: name = "CANCELLED"; break;
    case grpc::StatusCode::UNAVAILABLE: name = "UNAVAILABLE"; break;
    case grpc::StatusCode::DATA_LOSS: name = "DATA_LOSS"; break;
    default: break;
  }
  return std::string(name) + ": " + status.error_message();
}
bool Present(const std::string& value) {
  return !value.empty() && value.size() <= 128 &&
         value.find_first_not_of(" \t\r\n") != std::string::npos;
}
bool ValidTime(const Timestamp& value) {
  return value.seconds() >= -62135596800LL && value.seconds() <= 253402300799LL &&
         value.nanos() >= 0 && value.nanos() < 1000000000;
}
bool Before(const Timestamp& a, const Timestamp& b) {
  return a.seconds() < b.seconds() || (a.seconds() == b.seconds() && a.nanos() < b.nanos());
}
bool SameTime(const Timestamp& a, const Timestamp& b) {
  return a.seconds() == b.seconds() && a.nanos() == b.nanos();
}
bool ValidWeather(const flight::v1::WeatherSnapshot& weather,
                  const std::string& expected_kind, const Timestamp& cutoff) {
  if (weather.missing()) return true;
  static const std::regex airport("[A-Z0-9]{4}");
  return weather.kind() == expected_kind &&
      std::regex_match(weather.airport(), airport) && weather.has_issue_time() &&
      ValidTime(weather.issue_time()) && !Before(cutoff, weather.issue_time()) &&
      (!weather.has_report_age_minutes() ||
       (std::isfinite(weather.report_age_minutes()) && weather.report_age_minutes() >= 0));
}
bool ValidAirportWeather(const flight::v1::AirportWeather& weather,
                         const Timestamp& cutoff) {
  return (!weather.has_metar() || ValidWeather(weather.metar(), "METAR", cutoff)) &&
      (!weather.has_taf() || ValidWeather(weather.taf(), "TAF", cutoff));
}
bool ValidRequest(const flight::v1::PredictRequest& request) {
  if (!Present(request.trace_id()) || !request.has_flight()) return false;
  const auto& f = request.flight();
  static const std::regex airport("[A-Z0-9]{4}");
  const bool base_valid = Present(f.flight_id()) && Present(f.flight_number()) && Present(f.tail_number()) &&
      Present(f.aircraft_type()) && std::regex_match(f.departure_airport(), airport) &&
      std::regex_match(f.arrival_airport(), airport) && f.has_planned_off_block() &&
      f.has_planned_on_block() && ValidTime(f.planned_off_block()) &&
      ValidTime(f.planned_on_block()) && Before(f.planned_off_block(), f.planned_on_block()) &&
      f.departure_metar().size() <= 8192 && f.arrival_metar().size() <= 8192 &&
      (!f.has_planned_distance_miles() ||
       (std::isfinite(f.planned_distance_miles()) && f.planned_distance_miles() >= 0)) &&
      (!f.has_planned_flight_minutes() ||
       (std::isfinite(f.planned_flight_minutes()) && f.planned_flight_minutes() >= 0)) &&
      f.planned_takeoff_count() >= 0 && f.planned_landing_count() >= 0 &&
      f.planned_total_flow() >= 0;
  if (!base_valid || !request.has_zggg_context()) return base_valid;
  const auto& context = request.zggg_context();
  if (!context.has_prediction_cutoff() || !ValidTime(context.prediction_cutoff()) ||
      !SameTime(context.prediction_cutoff(), f.planned_off_block()) ||
      !Present(context.data_version()) ||
      !ValidAirportWeather(context.departure_weather(), context.prediction_cutoff()) ||
      !ValidAirportWeather(context.arrival_weather(), context.prediction_cutoff())) return false;
  std::unordered_set<unsigned> horizons;
  for (const auto& window : context.flow_features()) {
    if ((window.horizon_minutes() != 15 && window.horizon_minutes() != 30 &&
         window.horizon_minutes() != 60) || !horizons.insert(window.horizon_minutes()).second ||
        window.planned_takeoff() < 0 || window.planned_landing() < 0 ||
        window.completed_takeoff() < 0 || window.completed_landing() < 0) return false;
  }
  return true;
}
bool ValidResponse(const flight::v1::PredictResponse& response, const std::string& trace) {
  if (response.trace_id() != trace || !response.has_prediction() ||
      !Present(response.model_version()) || !Present(response.worker_id()) ||
      (response.source() != flight::v1::LLM && response.source() != flight::v1::TEST &&
       response.source() != flight::v1::BASELINE) ||
      response.inference_ms() < 0 || response.gateway_ms() < 0) return false;
  const auto& p = response.prediction();
  const bool prediction_valid = p.has_off_block() && p.has_takeoff() && p.has_landing() && p.has_on_block() &&
      ValidTime(p.off_block()) && ValidTime(p.takeoff()) && ValidTime(p.landing()) &&
      ValidTime(p.on_block()) && !Before(p.takeoff(), p.off_block()) &&
      !Before(p.landing(), p.takeoff()) && !Before(p.on_block(), p.landing());
  if (!prediction_valid || !response.has_airport_flow()) return prediction_valid;
  std::unordered_set<unsigned> horizons;
  for (const auto& window : response.airport_flow().windows()) {
    if ((window.horizon_minutes() != 15 && window.horizon_minutes() != 30 &&
         window.horizon_minutes() != 60) || !horizons.insert(window.horizon_minutes()).second ||
        window.takeoff() < 0 || window.landing() < 0 || window.total() < 0 ||
        window.total() != window.takeoff() + window.landing()) return false;
  }
  return response.airport_flow().model_version().size() <= 128 &&
      response.data_version().size() <= 128 && response.prompt_version().size() <= 128;
}
}  // namespace

grpc::Status GatewayService::Predict(grpc::ServerContext* upstream,
                                     const flight::v1::PredictRequest* request,
                                     flight::v1::PredictResponse* response) {
  if (!ValidRequest(*request))
    return {grpc::StatusCode::INVALID_ARGUMENT, "invalid flight request"};
  const auto started = std::chrono::steady_clock::now();
  const auto budget_end = std::min(upstream->deadline(), std::chrono::system_clock::now() +
      std::chrono::milliseconds(config_.rpc_timeout_ms));
  std::unordered_set<std::string> excluded;
  grpc::Status first_failure;
  bool has_first_failure = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (upstream->IsCancelled()) return {grpc::StatusCode::CANCELLED, "request cancelled"};
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        budget_end - std::chrono::system_clock::now());
    if (remaining.count() <= 0) return {grpc::StatusCode::DEADLINE_EXCEEDED, "request expired"};
    auto acquired = pool_.Acquire(excluded, CircuitBreaker::Clock::now());
    if (has_first_failure && acquired.kind != AcquireKind::kSelected) return first_failure;
    if (acquired.kind == AcquireKind::kAllEligibleFull)
      return {grpc::StatusCode::RESOURCE_EXHAUSTED, "all eligible workers are at capacity"};
    if (acquired.kind == AcquireKind::kNoEligibleNode)
      return {grpc::StatusCode::UNAVAILABLE, "no healthy worker available"};
    auto permit = std::move(acquired.permit);
    const auto node_id = permit->node_id();
    if (attempt == 1) permit->RecordFailover();
    auto downstream = grpc::ClientContext::FromServerContext(*upstream);
    downstream->set_deadline(budget_end);
    const auto attempt_start = std::chrono::steady_clock::now();
    auto status = permit->prediction_stub().Predict(downstream.get(), *request, response);
    const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - attempt_start).count();
    if (status.ok() && ValidResponse(*response, request->trace_id())) {
      LogAttempt(request->trace_id(), attempt + 1, node_id, attempt == 1,
                 static_cast<int>(grpc::StatusCode::OK), latency);
      permit->RecordSuccess(latency);
      permit->circuit_permit().OnPredictionSuccess();
      response->set_gateway_ms(std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started).count());
      return grpc::Status::OK;
    }
    if (status.ok()) status = {grpc::StatusCode::DATA_LOSS, "invalid worker prediction"};
    LogAttempt(request->trace_id(), attempt + 1, node_id, attempt == 1,
               static_cast<int>(status.error_code()), latency);
    response->Clear();
    permit->RecordFailure(std::to_string(static_cast<int>(status.error_code())),
                          StatusLabel(status), latency);
    if (status.error_code() == grpc::StatusCode::UNAVAILABLE)
      permit->circuit_permit().OnUnavailable(CircuitBreaker::Clock::now());
    else permit->circuit_permit().OnNeutralResult();
    excluded.insert(node_id);
    if (attempt == 0) {
      first_failure = status;
      has_first_failure = true;
    }
    const auto retry_remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        budget_end - std::chrono::system_clock::now());
    if (attempt == 0 && status.error_code() == grpc::StatusCode::UNAVAILABLE &&
        !upstream->IsCancelled() &&
        retry_remaining.count() >= config_.minimum_retry_budget_ms) continue;
    return status;
  }
  return {grpc::StatusCode::UNAVAILABLE, "failover exhausted"};
}
}  // namespace flight::gateway
