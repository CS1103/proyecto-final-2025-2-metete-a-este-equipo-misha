//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include "../algebra/tensor.h"
#include <cmath>

namespace utec::neural_network {

    /// Aplica una función callable elemento-wise a un tensor 2D
    /// \param tensor Tensor de entrada
    /// \param function Función a aplicar a cada elemento
    /// \return Nuevo tensor con la función aplicada
    template<typename T, typename F>
    utec::algebra::Tensor<T, 2> transform(
        const utec::algebra::Tensor<T, 2>& tensor,
        F&& function) {
        auto shape = tensor.shape();
        utec::algebra::Tensor<T, 2> output(shape[0], shape[1]);
        for (size_t row = 0; row < shape[0]; ++row) {
            for (size_t col = 0; col < shape[1]; ++col) {
                output(row, col) = function(tensor(row, col));
            }
        }
        return output;
    }

    /// Capa de activación ReLU: max(0, x)
    /// Introduce no-linealidad permitiendo que la red aprenda relaciones complejas
    template<typename T>
    class ReLU final : public ILayer<T> {
    private:
        utec::algebra::Tensor<T, 2> last_input_;

    public:
        /// Constructor por defecto
        ReLU() = default;

        /// Forward pass: aplica ReLU elemento-wise
        /// \param z Tensor de entrada
        /// \return Tensor con ReLU aplicado
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
            last_input_ = z;
            return transform(z,
                [](T x){ return x > T(0) ? x : T(0); });
        }

        /// Backward pass: calcula el gradiente de ReLU
        /// La derivada es 1 si x > 0, else 0
        /// \param gradient Gradiente de pérdida respecto a la salida
        /// \return Gradiente respecto a la entrada
        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
            auto shape = gradient.shape();
            utec::algebra::Tensor<T, 2> dx(shape[0], shape[1]);
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    dx(i, j) = (last_input_(i, j) > T(0)
                                ? gradient(i, j)
                                : T(0));
                }
            }
            return dx;
        }
    };

    /// Capa de activación Sigmoid: σ(x) = 1 / (1 + e^(-x))
    /// Mapea valores a [0, 1], útil para probabilidades en clasificación binaria
    template<typename T>
    class Sigmoid final : public ILayer<T> {
    private:
        utec::algebra::Tensor<T, 2> last_input_;

        /// Función de activación Sigmoid
        /// \param value Valor de entrada
        /// \return Valor de Sigmoid aplicado
        static T activate(T value) {
            return T(1) / (T(1) + std::exp(-value));
        }

    public:
        /// Constructor por defecto
        Sigmoid() = default;

        /// Forward pass: aplica Sigmoid elemento-wise
        /// \param z Tensor de entrada
        /// \return Tensor con Sigmoid aplicado
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
            last_input_ = z;
            return transform(z, activate);
        }

        /// Backward pass: calcula el gradiente de Sigmoid
        /// La derivada es σ(x) * (1 - σ(x))
        /// \param gradient Gradiente de pérdida respecto a la salida
        /// \return Gradiente respecto a la entrada
        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) override {
            auto shape = gradient.shape();
            utec::algebra::Tensor<T, 2> dx(shape[0], shape[1]);
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    T sigmoid_value = activate(last_input_(i, j));
                    dx(i, j) = gradient(i, j) * sigmoid_value * (T(1) - sigmoid_value);
                }
            }
            return dx;
        }
    };

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
