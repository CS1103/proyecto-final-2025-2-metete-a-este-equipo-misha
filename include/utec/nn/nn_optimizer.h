//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "nn_interfaces.h"
#include "../algebra/tensor.h"
#include <vector>
#include <cmath>

namespace utec::neural_network {

  /// Optimizador SGD (Stochastic Gradient Descent)
  /// Implementa: parameters = parameters - learning_rate * gradients
  ///
  /// Esta es la forma más simple de descenso de gradiente, donde la actualización
  /// de parámetros es proporcional solo al gradiente actual.
  template<typename T>
  class SGD final : public IOptimizer<T> {
  private:
    T learning_rate_;
    
  public:
    /// Constructor: inicializa SGD con una tasa de aprendizaje
    /// \param learning_rate Tasa de aprendizaje (típicamente 0.001 - 0.1)
    explicit SGD(T learning_rate = T(0.01))
      : learning_rate_(learning_rate) {}

    /// Actualiza los parámetros usando descenso de gradiente estocástico
    /// \param parameters Parámetros a actualizar (modificado in-place)
    /// \param gradients Gradientes de la función de pérdida respecto a los parámetros
    void update(utec::algebra::Tensor<T,2>& parameters,
                const utec::algebra::Tensor<T,2>& gradients) override
    {
      auto param_it = parameters.begin();
      auto grad_it = gradients.cbegin();
      for (; param_it != parameters.end(); ++param_it, ++grad_it) {
        *param_it -= learning_rate_ * (*grad_it);
      }
    }
  };

  /// Optimizador Adam (Adaptive Moment Estimation)
  ///
  /// Adam combina las ventajas de AdaGrad y RMSProp, manteniendo momentos de primer
  /// y segundo orden de los gradientes con corrección de sesgo. Es ampliamente utilizado
  /// en aplicaciones modernas de deep learning.
  ///
  /// Fórmulas:
  /// - m_t = β1 * m_{t-1} + (1-β1) * g_t  (momento de primer orden)
  /// - v_t = β2 * v_{t-1} + (1-β2) * g_t^2 (momento de segundo orden)
  /// - m̂_t = m_t / (1 - β1^t) (momento corregido)
  /// - v̂_t = v_t / (1 - β2^t) (momento corregido)
  /// - θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
  template<typename T>
  class Adam final : public IOptimizer<T> {
  private:
    T learning_rate_, beta1_, beta2_, epsilon_;
    size_t step_count_{0};
    utec::algebra::Tensor<T,2> first_moment_, second_moment_;
    bool initialized_{false};

    /// Inicialización perezosa del estado interno (momentos)
    void lazy_initialize(const utec::algebra::Tensor<T,2>& gradients) {
      if (!initialized_) {
        auto shape = gradients.shape();
        first_moment_ = utec::algebra::Tensor<T,2>(shape[0], shape[1]);
        second_moment_ = utec::algebra::Tensor<T,2>(shape[0], shape[1]);
        first_moment_.fill(T(0));
        second_moment_.fill(T(0));
        initialized_ = true;
      }
    }

  public:
    /// Constructor: inicializa Adam con los hiperparámetros
    ///
    /// \param learning_rate Tasa de aprendizaje (típicamente 0.0001 - 0.001)
    /// \param beta1 Coeficiente de decaimiento para el momento de primer orden (típicamente 0.9)
    /// \param beta2 Coeficiente de decaimiento para el momento de segundo orden (típicamente 0.999)
    /// \param epsilon Pequeño valor para evitar división por cero (típicamente 1e-8)
    explicit Adam(T learning_rate = T(0.001),
                  T beta1 = T(0.9),
                  T beta2 = T(0.999),
                  T epsilon = T(1e-8))
      : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon)
    {}

    /// Actualiza los parámetros usando el optimizador Adam
    /// \param parameters Parámetros a actualizar (modificado in-place)
    /// \param gradients Gradientes de la función de pérdida respecto a los parámetros
    void update(utec::algebra::Tensor<T,2>& parameters,
                const utec::algebra::Tensor<T,2>& gradients) override
    {
      lazy_initialize(gradients);
      ++step_count_;
      
      // Correcciones de sesgo para los momentos
      T bias_correction_1 = T(1) - std::pow(beta1_, T(step_count_));
      T bias_correction_2 = T(1) - std::pow(beta2_, T(step_count_));

      auto param_it = parameters.begin();
      auto grad_it = gradients.cbegin();
      auto m_it = first_moment_.begin();
      auto v_it = second_moment_.begin();
      
      for (; param_it != parameters.end(); ++param_it, ++grad_it, ++m_it, ++v_it) {
        // Actualiza momentos exponencialmente ponderados
        *m_it = beta1_ * (*m_it) + (T(1) - beta1_) * (*grad_it);
        *v_it = beta2_ * (*v_it) + (T(1) - beta2_) * (*grad_it) * (*grad_it);

        // Momentos corregidos por sesgo
        T m_hat = (*m_it) / bias_correction_1;
        T v_hat = (*v_it) / bias_correction_2;

        // Paso de actualización de parámetros
        *param_it -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
      }
    }

    /// Llamada después de procesar un paso (puede usarse para ajustes adicionales)
    void step() override {}
  };

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
