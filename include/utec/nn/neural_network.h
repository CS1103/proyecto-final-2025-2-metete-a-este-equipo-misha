//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_loss.h"
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>

namespace utec::neural_network {

/// Estructura para almacenar métricas de entrenamiento
template<typename T>
struct TrainingMetrics {
    size_t epochs_trained = 0;
    T final_loss = T(0);
    T best_loss = std::numeric_limits<T>::max();
    bool converged = false;
    std::vector<T> loss_history;
};

/// Estructura para almacenar métricas de evaluación
template<typename T>
struct EvaluationMetrics {
    T test_loss = T(0);
    T accuracy = T(0);
    T mean_absolute_error = T(0);
};

/// Red neuronal completa con soporte para entrenamiento y evaluación
///
/// Esta clase implementa una red neuronal profunda que permite:
/// - Agregar capas (Dense, Activaciones)
/// - Entrenar con descenso de gradiente y retropropagación
/// - Hacer predicciones
/// - Evaluar con múltiples métricas
template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;

    /// Realiza un paso forward a través de todas las capas
    algebra::Tensor<T,2> _forward_pass(const algebra::Tensor<T,2>& input) {
        auto output = input;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }

    /// Realiza un paso backward a través de todas las capas
    void _backward_pass(algebra::Tensor<T,2> gradient) {
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            gradient = (*it)->backward(gradient);
        }
    }

    /// Extrae un mini-batch de los datos
    std::pair<algebra::Tensor<T,2>, algebra::Tensor<T,2>>
    _get_batch(const algebra::Tensor<T,2>& X, const algebra::Tensor<T,2>& Y,
               size_t batch_start, size_t batch_size) {
        const size_t actual_size = std::min(batch_size, X.shape()[0] - batch_start);
        algebra::Tensor<T,2> X_batch(actual_size, X.shape()[1]);
        algebra::Tensor<T,2> Y_batch(actual_size, Y.shape()[1]);

        for (size_t i = 0; i < actual_size; ++i) {
            for (size_t j = 0; j < X.shape()[1]; ++j) {
                X_batch(i, j) = X(batch_start + i, j);
            }
            for (size_t j = 0; j < Y.shape()[1]; ++j) {
                Y_batch(i, j) = Y(batch_start + i, j);
            }
        }
        return {X_batch, Y_batch};
    }

public:
    /// Añade una capa a la red neuronal (propiedad única del puntero)
    /// \param layer Capa a agregar (Dense, ReLU, Sigmoid, etc.)
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.emplace_back(std::move(layer));
    }

    /// Realiza un paso forward a través de la red
    /// \param X Tensor de entrada [batch_size, num_features]
    /// \return Tensor de salida [batch_size, num_outputs]
    algebra::Tensor<T,2> forward(const algebra::Tensor<T,2>& X) {
        return _forward_pass(X);
    }

    /// Alias para forward (compatibilidad)
    algebra::Tensor<T,2> predict(const algebra::Tensor<T,2>& X) {
        return forward(X);
    }

    /// Entrena la red con los parámetros por defecto (MSELoss + SGD)
    /// Retorna el loss final del entrenamiento
    ///
    /// \param X Datos de entrada [num_samples, num_features]
    /// \param Y Datos de salida (etiquetas) [num_samples, num_outputs]
    /// \param epochs Número de épocas de entrenamiento
    /// \param learning_rate Tasa de aprendizaje del optimizador SGD
    /// \return Valor de pérdida final
    T train(const algebra::Tensor<T,2>& X,
            const algebra::Tensor<T,2>& Y,
            size_t epochs,
            T learning_rate) {
        const size_t total_samples = X.shape()[0];
        assert(total_samples == Y.shape()[0]);

        T final_loss = T(0);
        size_t batch_size = 32; // Batch size por defecto

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t batch_offset = 0; batch_offset < total_samples; batch_offset += batch_size) {
                auto [X_batch, Y_batch] = _get_batch(X, Y, batch_offset, batch_size);

                // Forward pass
                auto output = _forward_pass(X_batch);

                // Calcular pérdida con MSE
                MSELoss<T> loss_fn(output, Y_batch);
                final_loss = loss_fn.loss();
                auto gradient = loss_fn.loss_gradient();

                // Backward pass
                _backward_pass(gradient);

                // Actualizar parámetros con SGD
                SGD<T> optimizer(learning_rate);
                for (auto& layer : layers_) {
                    layer->update_params(optimizer);
                }
            }
        }

        return final_loss;
    }

    /// Entrena la red con soporte para early stopping y métricas avanzadas
    ///
    /// \param X Datos de entrada [num_samples, num_features]
    /// \param Y Datos de salida (etiquetas) [num_samples, num_outputs]
    /// \param max_epochs Número máximo de épocas
    /// \param learning_rate Tasa de aprendizaje inicial
    /// \param patience Número de épocas sin mejora antes de detener
    /// \param min_delta Mínima mejora considerada como progreso
    /// \return Estructura con métricas del entrenamiento
    TrainingMetrics<T> train_advanced(const algebra::Tensor<T,2>& X,
                                      const algebra::Tensor<T,2>& Y,
                                      size_t max_epochs,
                                      T learning_rate,
                                      size_t patience,
                                      T min_delta) {
        const size_t total_samples = X.shape()[0];
        assert(total_samples == Y.shape()[0]);

        TrainingMetrics<T> metrics;
        metrics.best_loss = std::numeric_limits<T>::max();
        metrics.converged = false;

        size_t batch_size = 32;
        size_t patience_counter = 0;

        for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
            T epoch_loss = T(0);
            size_t num_batches = 0;

            for (size_t batch_offset = 0; batch_offset < total_samples; batch_offset += batch_size) {
                auto [X_batch, Y_batch] = _get_batch(X, Y, batch_offset, batch_size);

                // Forward pass
                auto output = _forward_pass(X_batch);

                // Calcular pérdida con MSE
                MSELoss<T> loss_fn(output, Y_batch);
                epoch_loss += loss_fn.loss();
                auto gradient = loss_fn.loss_gradient();

                // Backward pass
                _backward_pass(gradient);

                // Actualizar parámetros con SGD
                SGD<T> optimizer(learning_rate);
                for (auto& layer : layers_) {
                    layer->update_params(optimizer);
                }

                num_batches++;
            }

            // Promedio del loss en la época
            epoch_loss /= static_cast<T>(num_batches);
            metrics.loss_history.push_back(epoch_loss);
            metrics.final_loss = epoch_loss;
            metrics.epochs_trained = epoch + 1;

            // Early stopping logic
            if (epoch_loss < metrics.best_loss - min_delta) {
                metrics.best_loss = epoch_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= patience) {
                    metrics.converged = true;
                    break;
                }
            }
        }

        return metrics;
    }

    /// Evalúa la red en datos de prueba
    ///
    /// \param X_test Datos de entrada de prueba [num_test_samples, num_features]
    /// \param Y_test Etiquetas de prueba [num_test_samples, num_outputs]
    /// \return Estructura con métricas de evaluación
    EvaluationMetrics<T> evaluate(const algebra::Tensor<T,2>& X_test,
                                  const algebra::Tensor<T,2>& Y_test) {
        EvaluationMetrics<T> metrics;

        // Forward pass
        auto predictions = forward(X_test);

        // Calcular loss
        MSELoss<T> loss_fn(predictions, Y_test);
        metrics.test_loss = loss_fn.loss();

        // Calcular MAE (Mean Absolute Error)
        T mae = T(0);
        for (size_t i = 0; i < X_test.shape()[0]; ++i) {
            for (size_t j = 0; j < Y_test.shape()[1]; ++j) {
                mae += std::abs(predictions(i, j) - Y_test(i, j));
            }
        }
        metrics.mean_absolute_error = mae / static_cast<T>(X_test.size());

        // Calcular accuracy (para clasificación binaria/multi-clase)
        size_t correct = 0;
        for (size_t i = 0; i < X_test.shape()[0]; ++i) {
            // Encontrar la clase predicha y real con mayor valor
            int pred_class = 0;
            int true_class = 0;
            T max_pred = predictions(i, 0);
            T max_true = Y_test(i, 0);

            for (size_t j = 1; j < Y_test.shape()[1]; ++j) {
                if (predictions(i, j) > max_pred) {
                    max_pred = predictions(i, j);
                    pred_class = j;
                }
                if (Y_test(i, j) > max_true) {
                    max_true = Y_test(i, j);
                    true_class = j;
                }
            }

            if (pred_class == true_class) {
                correct++;
            }
        }

        metrics.accuracy = static_cast<T>(correct) / static_cast<T>(X_test.shape()[0]);
        return metrics;
    }
};

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
