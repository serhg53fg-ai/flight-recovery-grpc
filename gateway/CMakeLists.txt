cmake_minimum_required(VERSION 3.16)
project(flight_gateway LANGUAGES CXX)
include(CTest)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Protobuf CONFIG QUIET)
if(NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf REQUIRED)
endif()
find_package(gRPC CONFIG QUIET)
if(NOT TARGET gRPC::grpc++)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
  add_library(gRPC::grpc++ INTERFACE IMPORTED)
  set_property(TARGET gRPC::grpc++ PROPERTY INTERFACE_LINK_LIBRARIES PkgConfig::GRPCPP)
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
  if(NOT GRPC_CPP_PLUGIN)
    message(FATAL_ERROR "grpc_cpp_plugin is required")
  endif()
  add_executable(gRPC::grpc_cpp_plugin IMPORTED)
  set_property(TARGET gRPC::grpc_cpp_plugin PROPERTY IMPORTED_LOCATION "${GRPC_CPP_PLUGIN}")
endif()
set(PROTO_OPTIONAL_FLAG)
if(Protobuf_VERSION VERSION_LESS "3.15")
  set(PROTO_OPTIONAL_FLAG --experimental_allow_proto3_optional)
endif()
find_package(Threads REQUIRED)
set(PROTO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../proto")
set(PROTO "${PROTO_ROOT}/flight/v1/prediction.proto")
set(HEALTH_PROTO "${PROTO_ROOT}/grpc/health/v1/health.proto")
set(GENERATED "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED}")
add_custom_command(
  OUTPUT "${GENERATED}/flight/v1/prediction.pb.cc" "${GENERATED}/flight/v1/prediction.pb.h"
         "${GENERATED}/flight/v1/prediction.grpc.pb.cc" "${GENERATED}/flight/v1/prediction.grpc.pb.h"
  COMMAND protobuf::protoc
  ARGS ${PROTO_OPTIONAL_FLAG} --proto_path=${PROTO_ROOT} --cpp_out=${GENERATED} --grpc_out=${GENERATED}
       --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin> ${PROTO}
  DEPENDS "${PROTO}" protobuf::protoc gRPC::grpc_cpp_plugin
  VERBATIM)
add_library(flight_proto STATIC "${GENERATED}/flight/v1/prediction.pb.cc"
                               "${GENERATED}/flight/v1/prediction.grpc.pb.cc")
target_include_directories(flight_proto PUBLIC "${GENERATED}")
target_link_libraries(flight_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
add_custom_command(
  OUTPUT "${GENERATED}/grpc/health/v1/health.pb.cc" "${GENERATED}/grpc/health/v1/health.pb.h"
         "${GENERATED}/grpc/health/v1/health.grpc.pb.cc" "${GENERATED}/grpc/health/v1/health.grpc.pb.h"
  COMMAND protobuf::protoc
  ARGS --proto_path=${PROTO_ROOT} --cpp_out=${GENERATED} --grpc_out=${GENERATED}
       --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin> ${HEALTH_PROTO}
  DEPENDS "${HEALTH_PROTO}" protobuf::protoc gRPC::grpc_cpp_plugin
  VERBATIM)
add_library(health_proto STATIC "${GENERATED}/grpc/health/v1/health.pb.cc"
                                "${GENERATED}/grpc/health/v1/health.grpc.pb.cc")
target_include_directories(health_proto PUBLIC "${GENERATED}")
target_link_libraries(health_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

add_executable(flight_gateway
  src/main.cc src/gateway_service.cc src/admin_service.cc src/health_monitor.cc
  src/backend_pool.cc src/metrics.cc src/config.cc src/circuit_breaker.cc src/gateway_logger.cc)
target_include_directories(flight_gateway PRIVATE src)
target_link_libraries(flight_gateway PRIVATE flight_proto health_proto Threads::Threads)
target_compile_options(flight_gateway PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_executable(gateway_core_test
    tests/gateway_core_test.cc
    src/config.cc
    src/circuit_breaker.cc
    src/metrics.cc
    src/backend_pool.cc
    src/gateway_logger.cc)
  target_include_directories(gateway_core_test PRIVATE src)
  target_link_libraries(gateway_core_test PRIVATE flight_proto health_proto protobuf::libprotobuf gRPC::grpc++ Threads::Threads)
  target_compile_options(gateway_core_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME gateway_core COMMAND gateway_core_test)
endif()
