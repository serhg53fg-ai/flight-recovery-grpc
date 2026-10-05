#pragma once
#include "backend_pool.h"
#include <flight/v1/prediction.grpc.pb.h>

namespace flight::gateway {
class AdminService final : public flight::v1::GatewayAdmin::Service {
 public:
  explicit AdminService(BackendPool& pool) : pool_(pool) {}
  grpc::Status GetClusterStatus(grpc::ServerContext*, const google::protobuf::Empty*,
                                flight::v1::ClusterStatus*) override;
 private:
  BackendPool& pool_;
};
}  // namespace flight::gateway
