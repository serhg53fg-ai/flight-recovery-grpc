#include "admin_service.h"
#include "config.h"
#include "gateway_service.h"
#include "health_monitor.h"
#include "gateway_logger.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <csignal>
#include <iostream>
#include <pthread.h>
#include <thread>

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--config")
      throw std::invalid_argument("usage: flight_gateway --config PATH");
    auto config = flight::gateway::LoadConfig(argv[2]);
    sigset_t signals;
    sigemptyset(&signals); sigaddset(&signals, SIGINT); sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);
    flight::gateway::BackendPool pool(config);
    flight::gateway::GatewayService gateway(pool, config);
    flight::gateway::AdminService admin(pool);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(1024 * 1024);
    builder.SetMaxSendMessageSize(1024 * 1024);
    int selected_port = 0;
    builder.AddListeningPort(config.listen, grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(&gateway);
    builder.RegisterService(&admin);
    auto server = builder.BuildAndStart();
    if (!server || selected_port == 0) throw std::runtime_error("gateway bind failed");
    auto* health = server->GetHealthCheckService();
    health->SetServingStatus(false);
    flight::gateway::HealthMonitor monitor(
        pool, std::chrono::milliseconds(config.health_interval_ms),
        std::chrono::milliseconds(config.health_timeout_ms),
        [health](bool serving) { health->SetServingStatus(serving); });
    monitor.Start();

    std::thread shutdown([&] {
      int signal = 0; sigwait(&signals, &signal);
      health->Shutdown();
      monitor.Stop();
      server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(1));
    });
    google::protobuf::Struct ready;
    (*ready.mutable_fields())["event"].set_string_value("gateway.ready");
    (*ready.mutable_fields())["listen"].set_string_value(config.listen);
    (*ready.mutable_fields())["workers"].set_number_value(config.workers.size());
    flight::gateway::WriteGatewayLog(ready);
    server->Wait();
    if (shutdown.joinable()) shutdown.join();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
