#pragma once
#include "backend_pool.h"
#include "config.h"
#include <flight/v1/prediction.grpc.pb.h>

namespace flight::gateway {
class GatewayService final : public flight::v1::PredictionGateway::Service {
 public:
  GatewayService(BackendPool& pool, const GatewayConfig& config) : pool_(pool), config_(config) {}
  grpc::Status Predict(grpc::ServerContext*, const flight::v1::PredictRequest*,
                       flight::v1::PredictResponse*) override;
 private:
  BackendPool& pool_;
  const GatewayConfig& config_;
};
}  // namespace flight::gateway
