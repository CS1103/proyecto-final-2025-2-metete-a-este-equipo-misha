#include "../include/utec/algebra/tensor.h"
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/nn_dense.h"
#include "../include/utec/nn/nn_activation.h"
#include "../include/utec/agent/PongAgent.h"
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

using namespace utec::algebra;
using namespace utec::neural_network;

/**
 * EJEMPLO: Entrenamiento de Red Neuronal para Pong
 * Este ejemplo muestra cómo entrenar una red para predecir movimientos en Pong
 */

// Generar datos sintéticos de Pong para entrenamiento
std::pair<Tensor<float,2>, Tensor<float,2>> generate_pong_data(size_t num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> vel_dist(-0.1f, 0.1f);

    // Estados: [ball_x, ball_y, ball_vx, ball_vy, paddle_y]
    Tensor<float,2> states(num_samples, 5);
    // Acciones: [-1, 0, 1] -> [up, stay, down] (one-hot encoded)
    Tensor<float,2> actions(num_samples, 3);

    for (size_t i = 0; i < num_samples; ++i) {
        float ball_x = pos_dist(gen);
        float ball_y = pos_dist(gen);
        float ball_vx = vel_dist(gen);
        float ball_vy = vel_dist(gen);
        float paddle_y = pos_dist(gen);

        // Estado actual
        states(i,0) = ball_x;
        states(i,1) = ball_y;
        states(i,2) = ball_vx;
        states(i,3) = ball_vy;
        states(i,4) = paddle_y;

        // Lógica simple para decidir acción óptima
        float future_ball_y = ball_y + ball_vy * (1.0f - ball_x) / std::abs(ball_vx + 1e-6f);

        actions(i,0) = 0; actions(i,1) = 0; actions(i,2) = 0; // Reset

        if (future_ball_y > paddle_y + 0.1f) {
            actions(i,2) = 1; // Move down
        } else if (future_ball_y < paddle_y - 0.1f) {
            actions(i,0) = 1; // Move up
        } else {
            actions(i,1) = 1; // Stay
        }
    }

    return {states, actions};
}

int main() {
    std::cout << "=== ENTRENAMIENTO PONG AGENT ===\n\n";

    // 1. GENERAR DATOS DE ENTRENAMIENTO
    std::cout << "Generando datos de entrenamiento...\n";
    auto [train_X, train_Y] = generate_pong_data(1000);
    auto [test_X, test_Y] = generate_pong_data(200);

    std::cout << "Datos generados:\n";
    std::cout << "- Entrenamiento: " << train_X.shape()[0] << " muestras\n";
    std::cout << "- Prueba: " << test_X.shape()[0] << " muestras\n";
    std::cout << "- Features: " << train_X.shape()[1] << " (ball_x, ball_y, ball_vx, ball_vy, paddle_y)\n";
    std::cout << "- Acciones: " << train_Y.shape()[1] << " (up, stay, down)\n\n";

    // 2. CREAR RED NEURONAL PARA PONG
    NeuralNetwork<float> pong_network;

    // Arquitectura más compleja para Pong: 5 -> 32 -> 16 -> 8 -> 3
    pong_network.add_layer(std::make_unique<Dense<float>>(5, 32));
    pong_network.add_layer(std::make_unique<ReLU<float>>());
    pong_network.add_layer(std::make_unique<Dense<float>>(32, 16));
    pong_network.add_layer(std::make_unique<ReLU<float>>());
    pong_network.add_layer(std::make_unique<Dense<float>>(16, 8));
    pong_network.add_layer(std::make_unique<ReLU<float>>());
    pong_network.add_layer(std::make_unique<Dense<float>>(8, 3));

    std::cout << "Red neuronal para Pong creada: 5->32->16->8->3\n\n";

    // 3. ENTRENAMIENTO CON VALIDACIÓN
    std::cout << "=== ENTRENAMIENTO CON VALIDACIÓN ===\n";

    auto training_metrics = pong_network.train_advanced(
        train_X, train_Y,
        500,        // epochs
        0.01f,      // learning rate (más bajo para estabilidad)
        20,         // patience
        1e-5f       // min_delta
    );

    std::cout << "\n=== RESULTADOS DEL ENTRENAMIENTO ===\n";
    std::cout << "Épocas: " << training_metrics.epochs_trained << "/500\n";
    std::cout << "Convergió: " << (training_metrics.converged ? "Sí" : "No") << "\n";
    std::cout << "Mejor loss: " << training_metrics.best_loss << "\n\n";

    // 4. EVALUACIÓN EN DATOS DE PRUEBA
    std::cout << "=== EVALUACIÓN EN DATOS DE PRUEBA ===\n";
    auto test_metrics = pong_network.evaluate(test_X, test_Y);

    std::cout << "Métricas de prueba:\n";
    std::cout << "- Loss: " << test_metrics.test_loss << "\n";
    std::cout << "- Accuracy: " << (test_metrics.accuracy * 100) << "%\n";
    std::cout << "- MAE: " << test_metrics.mean_absolute_error << "\n\n";

    // 5. ANÁLISIS DE PÉRDIDA DURANTE ENTRENAMIENTO
    std::cout << "=== EVOLUCIÓN DE LA PÉRDIDA ===\n";
    auto& loss_history = training_metrics.loss_history;

    // Mostrar pérdida cada 10% del entrenamiento
    size_t step = std::max(1UL, loss_history.size() / 10);
    for (size_t i = 0; i < loss_history.size(); i += step) {
        std::cout << "Época " << i << ": Loss = " << loss_history[i] << "\n";
    }
    if (!loss_history.empty()) {
        std::cout << "Época " << (loss_history.size()-1) << ": Loss = " << loss_history.back() << "\n";
    }
    std::cout << "\n";

    // 6. EJEMPLO DE PREDICCIÓN
    std::cout << "=== EJEMPLOS DE PREDICCIÓN ===\n";
    auto predictions = pong_network.forward(test_X);

    for (int i = 0; i < std::min(5, static_cast<int>(test_X.shape()[0])); ++i) {
        std::cout << "Ejemplo " << (i+1) << ":\n";
        std::cout << "  Estado: [ball_x=" << test_X(i,0)
                  << ", ball_y=" << test_X(i,1)
                  << ", paddle_y=" << test_X(i,4) << "]\n";

        // Encontrar acción real
        int real_action = 0;
        for (int j = 0; j < 3; ++j) {
            if (test_Y(i,j) > 0.5f) real_action = j;
        }

        // Encontrar acción predicha
        int pred_action = 0;
        float max_pred = predictions(i,0);
        for (int j = 1; j < 3; ++j) {
            if (predictions(i,j) > max_pred) {
                max_pred = predictions(i,j);
                pred_action = j;
            }
        }

        std::vector<std::string> action_names = {"UP", "STAY", "DOWN"};
        std::cout << "  Acción real: " << action_names[real_action] << "\n";
        std::cout << "  Acción predicha: " << action_names[pred_action];
        std::cout << " (" << (real_action == pred_action ? "✓" : "✗") << ")\n";
        std::cout << "  Confianza: [UP=" << predictions(i,0)
                  << ", STAY=" << predictions(i,1)
                  << ", DOWN=" << predictions(i,2) << "]\n\n";
    }

    return 0;
}
