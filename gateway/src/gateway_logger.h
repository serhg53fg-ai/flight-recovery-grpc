#pragma once
#include <google/protobuf/struct.pb.h>

namespace flight::gateway {
void WriteGatewayLog(const google::protobuf::Struct& record);
}
