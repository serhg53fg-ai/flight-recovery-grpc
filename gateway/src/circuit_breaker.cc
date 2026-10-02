#include "circuit_breaker.h"

#include <utility>

namespace flight::gateway {

CircuitBreaker::Permit::Permit(CircuitBreaker* owner, bool probe)
    : owner_(owner), probe_(probe) {}

CircuitBreaker::Permit::Permit(Permit&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), probe_(other.probe_) {}

CircuitBreaker::Permit& CircuitBreaker::Permit::operator=(Permit&& other) noexcept {
  if (this != &other) {
    Finish();
    owner_ = std::exchange(other.owner_, nullptr);
    probe_ = other.probe_;
  }
  return *this;
}

CircuitBreaker::Permit::~Permit() { Finish(); }

void CircuitBreaker::Permit::Finish() {
  if (owner_ != nullptr && probe_) owner_->ReleaseProbe();
  owner_ = nullptr;
}

void CircuitBreaker::Permit::OnUnavailable(TimePoint now) {
  if (owner_ != nullptr) owner_->RecordFailure(now);
  Finish();
}

void CircuitBreaker::Permit::OnPredictionSuccess() {
  if (owner_ != nullptr) owner_->PredictionSucceeded();
  Finish();
}

void CircuitBreaker::Permit::OnNeutralResult() { Finish(); }

CircuitBreaker::CircuitBreaker(std::uint32_t failure_threshold,
                               std::chrono::milliseconds open_cooldown)
    : failure_threshold_(failure_threshold), open_cooldown_(open_cooldown) {}

std::optional<CircuitBreaker::Permit> CircuitBreaker::TryAcquire(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == State::kOpen && now - opened_at_ >= open_cooldown_) {
    TransitionTo(State::kHalfOpen);
  }
  if (state_ == State::kOpen || (state_ == State::kHalfOpen && probe_active_)) {
    return std::nullopt;
  }
  const bool probe = state_ == State::kHalfOpen;
  if (probe) probe_active_ = true;
  return Permit(this, probe);
}

bool CircuitBreaker::CanAttempt(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == State::kOpen && now - opened_at_ >= open_cooldown_)
    TransitionTo(State::kHalfOpen);
  return state_ == State::kClosed ||
         (state_ == State::kHalfOpen && !probe_active_);
}

void CircuitBreaker::RecordFailure(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == State::kOpen) return;
  ++consecutive_failures_;
  if (state_ == State::kHalfOpen || consecutive_failures_ >= failure_threshold_) {
    TransitionTo(State::kOpen);
    opened_at_ = now;
    probe_active_ = false;
  }
}

void CircuitBreaker::OnUnavailable(TimePoint now) { RecordFailure(now); }
void CircuitBreaker::OnHealthFailure(TimePoint now) { RecordFailure(now); }
void CircuitBreaker::OnHealthSuccess() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == State::kClosed) consecutive_failures_ = 0;
}

void CircuitBreaker::PredictionSucceeded() {
  std::lock_guard<std::mutex> lock(mutex_);
  consecutive_failures_ = 0;
  TransitionTo(State::kClosed);
  probe_active_ = false;
}

void CircuitBreaker::ReleaseProbe() {
  std::lock_guard<std::mutex> lock(mutex_);
  probe_active_ = false;
}

CircuitBreaker::State CircuitBreaker::state(TimePoint now) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == State::kOpen && now - opened_at_ >= open_cooldown_) {
    TransitionTo(State::kHalfOpen);
  }
  return state_;
}

void CircuitBreaker::SetTransitionCallback(
    std::function<void(State, State)> callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  transition_callback_ = std::move(callback);
}

void CircuitBreaker::TransitionTo(State next) {
  if (state_ == next) return;
  const auto previous = state_;
  state_ = next;
  if (transition_callback_) transition_callback_(previous, next);
}

}  // namespace flight::gateway
