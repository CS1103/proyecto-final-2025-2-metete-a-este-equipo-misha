#include "../include/utec/algebra/tensor.h"
#include "../include/utec/nn/nn_activation.h"
#include "../include/utec/nn/nn_dense.h"
#include "../include/utec/nn/nn_loss.h"
#include "../include/utec/nn/neural_network.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace utec::algebra;
using namespace utec::neural_network;

void test_relu_forward_backward() {
    std::cout << "Test: ReLU forward/backward\n";
    using T = float;
    Tensor<T,2> M(2,2);
    M(0,0) = -1; M(0,1) = 2;
    M(1,0) = 0; M(1,1) = -3;

    auto relu = ReLU<T>();
    auto R = relu.forward(M);
    std::cout << "R(0,1) = " << R(0,1) << " (expected: 2)\n";
    assert(R(0,1) == 2);

    Tensor<T,2> GR(2,2);
    GR.fill(1.0f);
    auto dM = relu.backward(GR);
    std::cout << "dM(0,0) = " << dM(0,0) << ", dM(1,1) = " << dM(1,1) << "\n";
    assert(dM(0,0) == 0); // gradient is 0 where input was negative
    assert(dM(1,1) == 0); // gradient is 0 where input was negative
    std::cout << "✓ ReLU test passed\n\n";
}

void test_mse_loss() {
    std::cout << "Test: MSELoss forward/backward\n";
    using T = double;
    Tensor<T,2> P(1,2); P(0,0)=1; P(0,1)=2;
    Tensor<T,2> Tgt(1,2); Tgt(0,0)=0; Tgt(0,1)=4;

    auto loss = MSELoss<T>();
    T L = loss.forward(P, Tgt);
    std::cout << "Loss = " << L << " (expected: 2.5)\n";
    assert(std::abs(L - 2.5) < 1e-9);

    Tensor<T,2> dP = loss.backward();
    std::cout << "dP(0,1) = " << dP(0,1) << "\n";
    // Expected gradient: 2 * (pred - target) / num_elements = 2 * (2 - 4) / 2 = -2
    assert(std::abs(dP(0,1) - (-2.0)) < 1e-9);
    std::cout << "✓ MSE Loss test passed\n\n";
}

void test_dense_layer() {
    std::cout << "Test: Dense layer forward/backward\n";
    using T = float;

    Dense<T> dense(2, 3);
    Tensor<T,2> input(1, 2);
    input(0,0) = 1.0f;
    input(0,1) = 2.0f;

    auto output = dense.forward(input);
    assert(output.shape()[0] == 1);
    assert(output.shape()[1] == 3);
    std::cout << "✓ Dense forward shape correct: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n";

    Tensor<T,2> grad(1, 3);
    grad.fill(1.0f);
    auto input_grad = dense.backward(grad);
    assert(input_grad.shape()[0] == 1);
    assert(input_grad.shape()[1] == 2);
    std::cout << "✓ Dense backward shape correct: [" << input_grad.shape()[0] << ", " << input_grad.shape()[1] << "]\n\n";
}

void test_xor_training() {
    std::cout << "Test: XOR Training\n";
    using T = float;

    // XOR data
    Tensor<T,2> X(4,2);
    X(0,0)=0; X(0,1)=0; // input [0,0] -> output 0
    X(1,0)=0; X(1,1)=1; // input [0,1] -> output 1
    X(2,0)=1; X(2,1)=0; // input [1,0] -> output 1
    X(3,0)=1; X(3,1)=1; // input [1,1] -> output 0

    Tensor<T,2> Y(4,1);
    Y(0,0)=0; Y(1,0)=1; Y(2,0)=1; Y(3,0)=0;

    NeuralNetwork<T> net;
    net.add_layer(std::make_unique<Dense<T>>(2,4));
    net.add_layer(std::make_unique<ReLU<T>>());
    net.add_layer(std::make_unique<Dense<T>>(4,1));

    float final_loss = net.train(X, Y, 1000, 0.1f);
    std::cout << "Final loss: " << final_loss << "\n";

    // Test prediction after training
    auto predictions = net.forward(X);
    std::cout << "Predictions after training:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "[" << X(i,0) << "," << X(i,1) << "] -> " << predictions(i,0) << " (target: " << Y(i,0) << ")\n";
    }

    std::cout << "✓ XOR training completed\n\n";
}

void test_shape_mismatch() {
    std::cout << "Test: Shape mismatch exception\n";
    using T = float;

    NeuralNetwork<T> net;
    net.add_layer(std::make_unique<Dense<T>>(2,3));

    try {
        Tensor<T,2> wrong_input(1, 5); // Wrong input size
        net.forward(wrong_input);
        std::cout << "✗ Exception not thrown\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        std::cout << "✓ Exception thrown for shape mismatch\n\n";
    }
}

int main() {
    std::cout << "=== NEURAL NETWORK TESTS ===\n\n";

    test_relu_forward_backward();
    test_mse_loss();
    test_dense_layer();
    test_xor_training();
    test_shape_mismatch();

    std::cout << "All neural network tests completed! ✓\n";
    return 0;
}
