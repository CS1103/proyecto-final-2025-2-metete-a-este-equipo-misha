#pragma once

#include "utec/algebra/tensor.h"
#include "utec/nn/neural_network.h"
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>

namespace utec::agent {

struct State {
    float ball_x, ball_y;
    float paddle_y;
};

class EnvGym {
private:
    State current_state;
    float ball_vel_x, ball_vel_y;
    float paddle_vel;
    static constexpr float PADDLE_SPEED = 0.1f;
    static constexpr float BALL_SPEED = 0.05f;
    static constexpr float FIELD_WIDTH = 1.0f;
    static constexpr float FIELD_HEIGHT = 1.0f;
    static constexpr float PADDLE_HEIGHT = 0.2f;

public:
    EnvGym() {
        reset();
    }

    State reset() {
        current_state.ball_x = 0.5f;
        current_state.ball_y = 0.5f;
        current_state.paddle_y = 0.5f;

        // Random ball velocity
        ball_vel_x = BALL_SPEED * (rand() % 2 == 0 ? 1 : -1);
        ball_vel_y = BALL_SPEED * (rand() % 2 == 0 ? 1 : -1);

        return current_state;
    }

    State step(int action, float &reward, bool &done) {
        reward = 0.0f;
        done = false;

        // Update paddle position based on action
        if (action == -1) { // Move up
            current_state.paddle_y = std::max(0.0f, current_state.paddle_y - PADDLE_SPEED);
        } else if (action == 1) { // Move down
            current_state.paddle_y = std::min(1.0f - PADDLE_HEIGHT, current_state.paddle_y + PADDLE_SPEED);
        }
        // action == 0 means stay still

        // Update ball position
        current_state.ball_x += ball_vel_x;
        current_state.ball_y += ball_vel_y;

        // Ball collision with top and bottom walls
        if (current_state.ball_y <= 0.0f || current_state.ball_y >= 1.0f) {
            ball_vel_y = -ball_vel_y;
            current_state.ball_y = std::max(0.0f, std::min(1.0f, current_state.ball_y));
        }

        // Ball collision with left wall (player side - paddle at x=0.05)
        if (current_state.ball_x <= 0.05f) {
            // Check if paddle can hit the ball
            float paddle_top = current_state.paddle_y;
            float paddle_bottom = current_state.paddle_y + PADDLE_HEIGHT;

            if (current_state.ball_y >= paddle_top && current_state.ball_y <= paddle_bottom) {
                // Successful hit
                ball_vel_x = -ball_vel_x;
                current_state.ball_x = 0.05f;
                reward = 1.0f;
            } else {
                // Miss - game over
                reward = -1.0f;
                done = true;
            }
        }

        // Ball collision with right wall (opponent side)
        if (current_state.ball_x >= 0.95f) {
            ball_vel_x = -ball_vel_x;
            current_state.ball_x = 0.95f;
            reward = 0.5f; // Small reward for getting ball to opponent side
        }

        return current_state;
    }

    const State& get_state() const {
        return current_state;
    }
};

template<typename T>
class PongAgent {
private:
    std::unique_ptr<utec::neural_network::ILayer<T>> model;

public:
    explicit PongAgent(std::unique_ptr<utec::neural_network::ILayer<T>> m)
        : model(std::move(m)) {}

    // Convierte State a Tensor<T,2>, llama forward y devuelve -1/0/+1
    int act(const State &s) {
        // Convert state to tensor [1, 3] (batch_size=1, features=3)
        utec::algebra::Tensor<T, 2> input(1, 3);
        input(0, 0) = static_cast<T>(s.ball_x);
        input(0, 1) = static_cast<T>(s.ball_y);
        input(0, 2) = static_cast<T>(s.paddle_y);

        // Forward pass through the model
        auto output = model->forward(input);

        // Simple decision logic: if ball is above paddle, move up; if below, move down
        T ball_paddle_diff = static_cast<T>(s.ball_y) - static_cast<T>(s.paddle_y + 0.1f); // paddle center

        if (std::abs(ball_paddle_diff) < 0.05f) {
            return 0; // Stay still if close enough
        } else if (ball_paddle_diff > 0) {
            return 1; // Move down (ball is below)
        } else {
            return -1; // Move up (ball is above)
        }
    }
};

} // namespace utec::agent
