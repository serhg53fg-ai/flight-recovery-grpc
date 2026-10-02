#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <functional>

namespace flight::gateway {

class CircuitBreaker {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  enum class State { kClosed, kOpen, kHalfOpen };

  class Permit {
   public:
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;
    Permit(Permit&& other) noexcept;
    Permit& operator=(Permit&& other) noexcept;
    ~Permit();

    void OnUnavailable(TimePoint now);
    void OnPredictionSuccess();
    void OnNeutralResult();

   private:
    friend class CircuitBreaker;
    Permit(CircuitBreaker* owner, bool probe);
    void Finish();
    CircuitBreaker* owner_;
    bool probe_;
  };

  CircuitBreaker(std::uint32_t failure_threshold,
                 std::chrono::milliseconds open_cooldown);
  std::optional<Permit> TryAcquire(TimePoint now);
  bool CanAttempt(TimePoint now);
  void OnUnavailable(TimePoint now);
  void OnHealthFailure(TimePoint now);
  void OnHealthSuccess();
  State state(TimePoint now);
  void SetTransitionCallback(std::function<void(State, State)> callback);

 private:
  void RecordFailure(TimePoint now);
  void PredictionSucceeded();
  void ReleaseProbe();
  void TransitionTo(State state);

  std::mutex mutex_;
  const std::uint32_t failure_threshold_;
  const std::chrono::milliseconds open_cooldown_;
  State state_{State::kClosed};
  std::uint32_t consecutive_failures_{0};
  TimePoint opened_at_{};
  bool probe_active_{false};
  std::function<void(State, State)> transition_callback_;
};

}  // namespace flight::gateway
