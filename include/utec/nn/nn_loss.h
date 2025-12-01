//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "nn_interfaces.h"
#include "../algebra/tensor.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace utec::neural_network {

/// Pérdida de Error Cuadrático Medio (MSE)
/// Utilizada típicamente en problemas de regresión
/// L = (1/N) * Σ(y_pred - y_true)^2
template<typename T>
class MSELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> predictions_;  ///< Predicciones de la red
    utec::algebra::Tensor<T, 2> targets_;      ///< Valores reales (etiquetas)
    T cached_loss_{T(0)};

    /// Calcula la suma de todos los elementos de un tensor 2D
    static T sum_all(const utec::algebra::Tensor<T,2>& matrix) {
        return std::accumulate(matrix.cbegin(), matrix.cend(), T(0));
    }

public:
    /// Constructor: calcula el error cuadrático medio (MSE)
    /// \param y_pred Predicciones de la red [batch_size, num_outputs]
    /// \param y_true Valores objetivo (etiquetas) [batch_size, num_outputs]
    MSELoss(const utec::algebra::Tensor<T,2>& y_pred,
            const utec::algebra::Tensor<T,2>& y_true)
        : predictions_(y_pred), targets_(y_true) {
        auto difference = predictions_ - targets_;
        auto squared = difference * difference;
        auto total = sum_all(squared);
        auto num_elements = static_cast<T>(predictions_.size());
        cached_loss_ = total / num_elements;
    }

    /// Retorna el valor de la pérdida (MSE)
    T loss() const override {
        return cached_loss_;
    }

    /// Retorna el gradiente: dL/dŷ = 2*(y_pred - y_true)/N
    utec::algebra::Tensor<T,2> loss_gradient() const override {
        auto difference = predictions_ - targets_;
        auto num_elements = static_cast<T>(predictions_.size());
        return difference * (T(2) / num_elements);
    }
};

/// Pérdida de Entropía Cruzada Binaria (BCE)
/// Utilizada en problemas de clasificación binaria
/// L = -(1/N) * Σ[y*log(p) + (1-y)*log(1-p)]
template<typename T>
class BCELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> predictions_;
    utec::algebra::Tensor<T, 2> targets_;
    T cached_loss_{T(0)};
    static constexpr T epsilon = T(1e-7);  ///< Pequeño valor para evitar log(0)

    /// Realiza clamp para estabilidad numérica
    static T safe_clip(T probability) {
        if (probability < epsilon)         
            return epsilon;
        if (probability > T(1) - epsilon)  
            return T(1) - epsilon;
        return probability;
    }

public:
    /// Constructor: calcula la entropía cruzada binaria (BCE)
    /// \param y_pred Predicciones de la red [batch_size, num_outputs]
    /// \param y_true Valores objetivo (etiquetas) [batch_size, num_outputs]
    BCELoss(const utec::algebra::Tensor<T,2>& y_pred,
            const utec::algebra::Tensor<T,2>& y_true)
        : predictions_(y_pred), targets_(y_true) {
        auto shape = predictions_.shape();
        T accumulated_loss = T(0);
        for (size_t row = 0; row < shape[0]; ++row) {
            for (size_t col = 0; col < shape[1]; ++col) {
                T predicted_probability = safe_clip(predictions_(row, col));
                T target = targets_(row, col);
                accumulated_loss += -target * std::log(predicted_probability)
                                 - (T(1) - target) * std::log(T(1) - predicted_probability);
            }
        }
        auto num_elements = static_cast<T>(predictions_.size());
        cached_loss_ = accumulated_loss / num_elements;
    }

    /// Retorna el valor de la pérdida (BCE)
    T loss() const override {
        return cached_loss_;
    }

    /// Retorna el gradiente: dL/dŷ = (p - y)/(p*(1-p)*N)
    utec::algebra::Tensor<T,2> loss_gradient() const override {
        auto shape = predictions_.shape();
        utec::algebra::Tensor<T,2> gradient(shape[0], shape[1]);
        T num_elements = static_cast<T>(predictions_.size());
        for (size_t row = 0; row < shape[0]; ++row) {
            for (size_t col = 0; col < shape[1]; ++col) {
                T predicted_probability = safe_clip(predictions_(row, col));
                T target = targets_(row, col);
                gradient(row, col) = (predicted_probability - target) 
                                   / (predicted_probability * (T(1) - predicted_probability) * num_elements);
            }
        }
        return gradient;
    }
};

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
