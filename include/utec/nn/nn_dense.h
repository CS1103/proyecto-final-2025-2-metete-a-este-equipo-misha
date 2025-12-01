//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include "../algebra/tensor.h"
#include <random>
#include <cmath>

namespace utec::neural_network {

template<typename T>
class Dense final : public ILayer<T> {
private:
    algebra::Tensor<T, 2> weights_;
    algebra::Tensor<T, 2> biases_;
    algebra::Tensor<T, 2> input_;
    algebra::Tensor<T, 2> grad_w_;
    algebra::Tensor<T, 2> grad_b_;

    /// Inicialización Xavier: uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
    static void xavier_init(algebra::Tensor<T, 2>& tensor) {
        auto shape = tensor.shape();
        size_t fan_in = shape[0];
        size_t fan_out = shape[1];
        T limit = std::sqrt(T(6) / (T(fan_in) + T(fan_out)));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-limit, limit);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                tensor(i, j) = dist(gen);
            }
        }
    }

    /// Inicialización de sesgos a cero
    static void zero_init(algebra::Tensor<T, 2>& tensor) {
        tensor.fill(T(0));
    }

public:
    /// Constructor por defecto (parámetros no inicializados - no recomendado usar sin configurar)
    Dense()
      : weights_(0, 0), biases_(0, 0),
        input_(0, 0), grad_w_(0, 0), grad_b_(0, 0)
    {}

    /// Constructor: inicializa una capa densa con dimensiones especificadas
    /// Utiliza inicialización Xavier para los pesos y ceros para los sesgos
    Dense(size_t input_features, size_t output_features)
      : weights_(input_features, output_features),
        biases_(1, output_features),
        input_(1, input_features),
        grad_w_(input_features, output_features),
        grad_b_(1, output_features)
    {
        xavier_init(weights_);
        zero_init(biases_);
    }

    /// Constructor: inicializa una capa densa con functores personalizados
    /// init_w y init_b son functores que inicializan los pesos y sesgos respectivamente
    template<typename InitWFun, typename InitBFun>
    Dense(size_t input_features,
          size_t output_features,
          InitWFun initialize_weights,
          InitBFun initialize_biases)
      : weights_(input_features, output_features),
        biases_(1, output_features),
        input_(1, input_features),
        grad_w_(input_features, output_features),
        grad_b_(1, output_features)
    {
        // Inicializa pesos y sesgos usando los functores proporcionados
        initialize_weights(weights_);
        initialize_biases(biases_);
    }

    /// Propagación hacia adelante: Z = X * W + b
    ///
    /// \param x Tensor de entrada [batch_size, input_features]
    /// \return Tensor de salida [batch_size, output_features]
    algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2>& x) override {
        input_ = x;
        // Z = X · W  → shape [batch, output_features]
        auto Z = algebra::matrix_product(x, weights_);
        // Suma el sesgo con broadcasting
        return Z + biases_;
    }

    /// Propagación hacia atrás: calcula gradientes respecto a pesos, sesgos y entrada
    ///
    /// \param dZ Gradiente de pérdida respecto a la salida [batch_size, output_features]
    /// \return Gradiente respecto a la entrada [batch_size, input_features]
    algebra::Tensor<T, 2> backward(const algebra::Tensor<T, 2>& dZ) override {
        // Gradiente respecto a pesos: grad_w = Xᵀ · dZ
        // Dimensiones: [input_features, batch] · [batch, output_features]
        grad_w_ = algebra::matrix_product(algebra::transpose(input_), dZ);

        // Gradiente respecto a sesgos: grad_b = ones_row · dZ
        // Dimensiones: [1, batch] · [batch, output_features] = [1, output_features]
        const auto batch = dZ.shape()[0];
        algebra::Tensor<T, 2> ones(1, batch);
        ones.fill(T(1));
        grad_b_ = algebra::matrix_product(ones, dZ);

        // Retorna el gradiente respecto a la entrada
        return algebra::matrix_product(dZ, algebra::transpose(weights_));
    }

    /// Actualiza los parámetros (pesos y sesgos) usando el optimizador proporcionado
    ///
    /// \param optimizer Optimizador a utilizar (SGD, Adam, etc.)
    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights_, grad_w_);
        optimizer.update(biases_, grad_b_);
        optimizer.step();
    }
};

}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
