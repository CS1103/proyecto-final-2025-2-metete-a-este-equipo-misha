#include "../include/utec/algebra/tensor.h"
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/nn_dense.h"
#include "../include/utec/nn/nn_activation.h"
#include <iostream>
#include <iomanip>

using namespace utec::algebra;
using namespace utec::neural_network;

int main() {
    std::cout << "=== ENTRENAMIENTO DE RED NEURONAL - XOR ===\n\n";

    // 1. PREPARAR LOS DATOS
    // Datos de entrada XOR: [0,0], [0,1], [1,0], [1,1]
    Tensor<float, 2> X(4, 2);
    X(0,0) = 0; X(0,1) = 0;  // [0,0] -> 0
    X(1,0) = 0; X(1,1) = 1;  // [0,1] -> 1
    X(2,0) = 1; X(2,1) = 0;  // [1,0] -> 1
    X(3,0) = 1; X(3,1) = 1;  // [1,1] -> 0

    // Salidas esperadas
    Tensor<float, 2> Y(4, 1);
    Y(0,0) = 0;  // 0 XOR 0 = 0
    Y(1,0) = 1;  // 0 XOR 1 = 1
    Y(2,0) = 1;  // 1 XOR 0 = 1
    Y(3,0) = 0;  // 1 XOR 1 = 0

    std::cout << "Datos de entrenamiento creados:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "Input: [" << X(i,0) << ", " << X(i,1)
                  << "] -> Output: " << Y(i,0) << "\n";
    }
    std::cout << "\n";

    // 2. CREAR LA RED NEURONAL
    NeuralNetwork<float> network;

    // Arquitectura: 2 -> 4 -> 4 -> 1 (con activaciones ReLU)
    network.add_layer(std::make_unique<Dense<float>>(2, 4));
    network.add_layer(std::make_unique<ReLU<float>>());
    network.add_layer(std::make_unique<Dense<float>>(4, 4));
    network.add_layer(std::make_unique<ReLU<float>>());
    network.add_layer(std::make_unique<Dense<float>>(4, 1));

    std::cout << "Red neuronal creada: 2->4->4->1\n\n";

    // 3. ENTRENAMIENTO BÁSICO
    std::cout << "=== ENTRENAMIENTO BÁSICO ===\n";
    float learning_rate = 0.1f;
    size_t epochs = 1000;

    float final_loss = network.train(X, Y, epochs, learning_rate);
    std::cout << "Entrenamiento completado!\n";
    std::cout << "Loss final: " << std::scientific << final_loss << "\n\n";

    // 4. ENTRENAMIENTO AVANZADO CON MÉTRICAS
    std::cout << "=== ENTRENAMIENTO AVANZADO ===\n";

    // Crear nueva red para el entrenamiento avanzado
    NeuralNetwork<float> advanced_network;
    advanced_network.add_layer(std::make_unique<Dense<float>>(2, 8));
    advanced_network.add_layer(std::make_unique<ReLU<float>>());
    advanced_network.add_layer(std::make_unique<Dense<float>>(8, 4));
    advanced_network.add_layer(std::make_unique<ReLU<float>>());
    advanced_network.add_layer(std::make_unique<Dense<float>>(4, 1));

    // Entrenamiento con early stopping y adaptive learning rate
    auto metrics = advanced_network.train_advanced(
        X, Y,           // datos
        2000,           // max epochs
        0.2f,           // learning rate inicial
        50,             // patience para early stopping
        1e-6f           // min_delta para improvement
    );

    std::cout << "\n=== RESULTADOS DEL ENTRENAMIENTO AVANZADO ===\n";
    std::cout << "Épocas entrenadas: " << metrics.epochs_trained << "/2000\n";
    std::cout << "Convergió: " << (metrics.converged ? "Sí" : "No") << "\n";
    std::cout << "Mejor loss: " << std::scientific << metrics.best_loss << "\n";
    std::cout << "Loss final: " << std::scientific << metrics.final_loss << "\n\n";

    // 5. EVALUACIÓN DEL MODELO
    std::cout << "=== EVALUACIÓN DEL MODELO ===\n";
    auto evaluation = advanced_network.evaluate(X, Y);

    std::cout << "Métricas de evaluación:\n";
    std::cout << "- Test Loss: " << std::scientific << evaluation.test_loss << "\n";
    std::cout << "- Accuracy: " << std::fixed << std::setprecision(2)
              << (evaluation.accuracy * 100) << "%\n";
    std::cout << "- MAE: " << std::scientific << evaluation.mean_absolute_error << "\n\n";

    // 6. PREDICCIONES
    std::cout << "=== PREDICCIONES ===\n";
    auto predictions = advanced_network.forward(X);

    std::cout << "Input -> Esperado vs Predicción\n";
    for (int i = 0; i < 4; ++i) {
        float pred = predictions(i, 0);
        float expected = Y(i, 0);
        float error = std::abs(pred - expected);
        std::cout << "[" << X(i,0) << "," << X(i,1) << "] -> "
                  << std::fixed << std::setprecision(4)
                  << "Predicho: " << pred
                  << ", Esperado: " << expected
                  << ", Error: " << error << "\n";
    }

    std::cout << "\n=== ENTRENAMIENTO COMPLETADO ===\n";
    return 0;
}
                  << expected << " vs " << std::fixed << std::setprecision(4) << pred;

        // Clasificación binaria (threshold = 0.5)
        bool correct = (pred > 0.5f) == (expected > 0.5f);
        std::cout << " (" << (correct ? "✓" : "✗") << ")\n";
    }

    return 0;
}
