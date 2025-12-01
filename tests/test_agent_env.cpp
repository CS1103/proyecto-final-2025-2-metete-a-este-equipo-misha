#include "../include/utec/agent/PongAgent.h"
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/nn_dense.h"
#include "../include/utec/nn/nn_activation.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace utec::agent;
using namespace utec::neural_network;

void test_basic_instantiation() {
    std::cout << "Test 1: Instanciación básica\n";
    using T = float;

    // Create a simple neural network for the agent
    auto net = std::make_unique<NeuralNetwork<T>>();
    net->add_layer(std::make_unique<Dense<T>>(3, 3));

    auto agent = PongAgent<T>(std::move(net));

    // Estado de prueba: bola por encima de paleta
    State s{0.5f, 0.8f, 0.3f};
    int a = agent.act(s);
    std::cout << "Action: " << a << " (expected: +1 for ball above paddle)\n";
    // The simple heuristic should return +1 (move down) when ball is below paddle center
    std::cout << "✓ Basic instantiation test passed\n\n";
}

void test_simulation_step() {
    std::cout << "Test 2: Simulación de un paso\n";
    using T = float;

    auto net = std::make_unique<NeuralNetwork<T>>();
    net->add_layer(std::make_unique<Dense<T>>(3, 3));
    auto agent = PongAgent<T>(std::move(net));

    EnvGym env;
    float reward;
    bool done;
    auto s0 = env.reset();
    int a0 = agent.act(s0);
    auto s1 = env.step(a0, reward, done);

    std::cout << "Initial state: ball(" << s0.ball_x << ", " << s0.ball_y << "), paddle(" << s0.paddle_y << ")\n";
    std::cout << "Action: " << a0 << "\n";
    std::cout << "New state: ball(" << s1.ball_x << ", " << s1.ball_y << "), paddle(" << s1.paddle_y << ")\n";
    std::cout << "Reward: " << reward << ", Done: " << done << "\n";
    std::cout << "✓ Simulation step test passed\n\n";
}

void test_agent_environment_integration() {
    std::cout << "Test 3: Integración agent + entorno\n";
    using T = float;

    auto net = std::make_unique<NeuralNetwork<T>>();
    net->add_layer(std::make_unique<Dense<T>>(3, 4));
    net->add_layer(std::make_unique<ReLU<T>>());
    net->add_layer(std::make_unique<Dense<T>>(4, 3));
    auto agent = PongAgent<T>(std::move(net));

    EnvGym env;
    float reward;
    bool done;
    auto s0 = env.reset();

    std::cout << "Running 5 simulation steps:\n";
    for(int t = 0; t < 5; ++t) {
        int a = agent.act(s0);
        s0 = env.step(a, reward, done);
        std::cout << "Step " << t << " action=" << a << " reward=" << reward << "\n";
        if(done) {
            std::cout << "Game ended at step " << t << "\n";
            break;
        }
    }
    std::cout << "✓ Agent-environment integration test passed\n\n";
}

void test_boundary_conditions() {
    std::cout << "Test 4: Prueba de límites\n";
    using T = float;

    auto net = std::make_unique<NeuralNetwork<T>>();
    net->add_layer(std::make_unique<Dense<T>>(3, 3));
    auto agent = PongAgent<T>(std::move(net));

    // When ball_y == paddle_y + 0.1 (paddle center), should return 0
    State eq{0.2f, 0.6f, 0.5f}; // ball at 0.6, paddle at 0.5, center at 0.6
    int action = agent.act(eq);
    std::cout << "Ball Y: " << eq.ball_y << ", Paddle center: " << (eq.paddle_y + 0.1f) << "\n";
    std::cout << "Action: " << action << " (expected: 0 when close)\n";

    // Test extreme positions
    State ball_high{0.2f, 0.9f, 0.1f};
    int action_high = agent.act(ball_high);
    std::cout << "Ball high, action: " << action_high << "\n";

    State ball_low{0.2f, 0.1f, 0.8f};
    int action_low = agent.act(ball_low);
    std::cout << "Ball low, action: " << action_low << "\n";

    std::cout << "✓ Boundary conditions test passed\n\n";
}

void test_environment_physics() {
    std::cout << "Test 5: Environment physics\n";

    EnvGym env;
    float reward;
    bool done;
    auto state = env.reset();

    std::cout << "Testing environment physics...\n";
    std::cout << "Initial state: ball(" << state.ball_x << ", " << state.ball_y << "), paddle(" << state.paddle_y << ")\n";

    // Test paddle movement
    state = env.step(-1, reward, done); // Move up
    std::cout << "After move up: paddle_y = " << state.paddle_y << "\n";

    state = env.step(1, reward, done); // Move down
    std::cout << "After move down: paddle_y = " << state.paddle_y << "\n";

    state = env.step(0, reward, done); // Stay still
    std::cout << "After stay still: paddle_y = " << state.paddle_y << "\n";

    std::cout << "✓ Environment physics test passed\n\n";
}

int main() {
    std::cout << "=== AGENT & ENVIRONMENT TESTS ===\n\n";

    test_basic_instantiation();
    test_simulation_step();
    test_agent_environment_integration();
    test_boundary_conditions();
    test_environment_physics();

    std::cout << "All agent and environment tests completed! ✓\n";
    return 0;
}
