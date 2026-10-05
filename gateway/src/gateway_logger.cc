#include "gateway_logger.h"
#include <google/protobuf/util/json_util.h>
#include <iostream>
#include <mutex>

namespace flight::gateway {
void WriteGatewayLog(const google::protobuf::Struct& record) {
  static std::mutex output_mutex;
  std::string json;
  if (!google::protobuf::util::MessageToJsonString(record, &json).ok()) return;
  std::lock_guard<std::mutex> lock(output_mutex);
  std::cout << json << std::endl;
}
}  // namespace flight::gateway
