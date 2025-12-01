//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <array>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstddef>
#include <ostream>

namespace utec::algebra {

template<typename T, size_t N>
class Tensor {
private:
    std::array<size_t, N> dims_;
    std::vector<T> data_;

public:
    // Constructor: inicializa el tensor con las dimensiones especificadas
    template<typename... SizeArgs,
             typename = std::enable_if_t<(sizeof...(SizeArgs) == N) &&
                        (std::is_convertible_v<SizeArgs, size_t> && ...)>>
    explicit Tensor(SizeArgs... dims) {
        std::array<size_t, N> dimensions{ static_cast<size_t>(dims)... };
        dims_ = dimensions;
        size_t total_elements = 1;
        for (auto dimension : dims_) {
            total_elements *= dimension;
        }
        data_.assign(total_elements, T());
    }

    // Retorna la forma (dimensiones) del tensor
    std::array<size_t, N> shape() const noexcept { 
        return dims_; 
    }
    
    // Retorna el número total de elementos en el tensor
    size_t size() const noexcept {
        return std::accumulate(dims_.begin(), dims_.end(), size_t(1), std::multiplies<>());
    }

    // Rellena todos los elementos con el valor especificado
    void fill(const T& value) { 
        std::fill(data_.begin(), data_.end(), value); 
    }

    // Cambia las dimensiones del tensor (redimensiona si es necesario)
    template<typename... SizeArgs,
             typename = std::enable_if_t<(std::is_convertible_v<SizeArgs, size_t> && ...)>>
    void reshape(SizeArgs... dims) {
        std::array<size_t, N> new_dimensions{ static_cast<size_t>(dims)... };
        size_t new_total = 1;
        for (auto dimension : new_dimensions) {
            new_total *= dimension;
        }
        if (new_total != size()) {
            data_.resize(new_total);
        }
        dims_ = new_dimensions;
    }

    // Asigna valores desde una lista de inicialización
    Tensor& operator=(std::initializer_list<T> list) {
        if (list.size() != data_.size()) {
            throw std::runtime_error("El tamaño de la lista no coincide con el tensor");
        }
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    // Acceso a elementos usando indexación multi-dimensional (row-major)
    template<typename... Idx,
             typename = std::enable_if_t<sizeof...(Idx) == N &&
                        (std::is_convertible_v<Idx, size_t> && ...)>>
    T& operator()(Idx... idx) {
        std::array<size_t, N> indices{ size_t(idx)... };
        size_t offset = 0, stride = 1;
        for (int k = N - 1; k >= 0; --k) {
            offset += indices[k] * stride;
            stride *= dims_[k];
        }
        return data_[offset];
    }
    
    // Versión const del operador de acceso
    template<typename... Idx>
    const T& operator()(Idx... idx) const { 
        return const_cast<Tensor&>(*this)(idx...); 
    }

    // Iteradores para acceso secuencial a los datos
    auto begin()        { return data_.begin(); }
    auto end()          { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend()   const { return data_.cend(); }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if constexpr (N == 1) {
            for (size_t i = 0; i < t.dims_[0]; ++i) {
                os << t.data_[i] << (i + 1 < t.dims_[0] ? " " : "");
            }
        } else if constexpr (N == 2) {
            os << "{\n";
            size_t R = t.dims_[0], C = t.dims_[1];
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    os << t.data_[i * C + j]
                       << (j + 1 < C ? " " : "");
                }
                os << "\n";
            }
            os << "}";
        } else if constexpr (N == 3 || N == 4) {
            // Simplificación: reutiliza flatten para N>2
            auto flat = t.data_;
            os << "{ ";
            for (size_t i = 0; i < flat.size(); ++i) {
                os << flat[i] << (i + 1 < flat.size() ? ", " : " ");
            }
            os << "}";
        } else {
            os << "{ ";
            for (size_t i = 0; i < t.data_.size(); ++i) {
                os << t.data_[i] << (i + 1 < t.data_.size() ? ", " : " ");
            }
            os << "}";
        }
        return os;
    }

    // Broadcasting +, -, * entre tensores 2D y 3D
    Tensor operator+(const Tensor& o) const {
        auto A = shape(), B = o.shape();
        if constexpr (N == 2) {
            size_t R = std::max(A[0], B[0]);
            size_t C = std::max(A[1], B[1]);
            Tensor result(R, C);
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    T a = (*this)(A[0] == 1 ? 0 : i, A[1] == 1 ? 0 : j);
                    T b = o      (B[0] == 1 ? 0 : i, B[1] == 1 ? 0 : j);
                    result(i,j) = a + b;
                }
            }
            return result;
        }
        else if constexpr (N == 3) {
            size_t D0 = std::max(A[0], B[0]);
            size_t D1 = std::max(A[1], B[1]);
            size_t D2 = std::max(A[2], B[2]);
            Tensor result(D0, D1, D2);
            for (size_t i = 0; i < D0; ++i)
            for (size_t j = 0; j < D1; ++j)
            for (size_t k = 0; k < D2; ++k) {
                T a = (*this)(
                    A[0]==1?0:i,
                    A[1]==1?0:j,
                    A[2]==1?0:k);
                T b = o(
                    B[0]==1?0:i,
                    B[1]==1?0:j,
                    B[2]==1?0:k);
                result(i,j,k) = a + b;
            }
            return result;
        }
        else {
            throw std::runtime_error("Broadcast no implementado para N>3");
        }
    }

    Tensor operator-(const Tensor& o) const { return *this + (o * T(-1)); }
    Tensor operator*(const Tensor& o) const {
        // Element-wise
        auto A = shape(), B = o.shape();
        if constexpr (N == 2) {
            size_t R = std::max(A[0], B[0]);
            size_t C = std::max(A[1], B[1]);
            Tensor result(R, C);
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < C; ++j) {
                    T a = (*this)(A[0] == 1 ? 0 : i, A[1] == 1 ? 0 : j);
                    T b = o      (B[0] == 1 ? 0 : i, B[1] == 1 ? 0 : j);
                    result(i,j) = a * b;
                }
            }
            return result;
        }
        else if constexpr (N == 3) {
            size_t D0 = std::max(A[0], B[0]);
            size_t D1 = std::max(A[1], B[1]);
            size_t D2 = std::max(A[2], B[2]);
            Tensor result(D0, D1, D2);
            for (size_t i = 0; i < D0; ++i)
            for (size_t j = 0; j < D1; ++j)
            for (size_t k = 0; k < D2; ++k) {
                T a = (*this)(
                    A[0]==1?0:i,
                    A[1]==1?0:j,
                    A[2]==1?0:k);
                T b = o(
                    B[0]==1?0:i,
                    B[1]==1?0:j,
                    B[2]==1?0:k);
                result(i,j,k) = a * b;
            }
            return result;
        }
        else {
            throw std::runtime_error("Broadcast no implementado para N>3");
        }
    }

    // Escalar ops
    Tensor operator*(T scalar) const {
        Tensor r = *this;
        for (auto& v : r.data_) v *= scalar;
        return r;
    }
    Tensor operator/(T scalar) const {
        Tensor r = *this;
        for (auto& v : r.data_) v /= scalar;
        return r;
    }
    Tensor operator+(T scalar) const {
        Tensor r = *this;
        for (auto& v : r.data_) v += scalar;
        return r;
    }
    friend Tensor operator+(T scalar, const Tensor& t) { return t + scalar; }
    Tensor() noexcept: dims_{}, data_{} {}
};


    // Transpuesta
    template<typename T>
    Tensor<T,2> transpose(const Tensor<T,2>& m) {
        auto s = m.shape();
        Tensor<T,2> out(s[1], s[0]);
        for (size_t i = 0; i < s[0]; ++i)
            for (size_t j = 0; j < s[1]; ++j)
                out(j,i) = m(i,j);
        return out;
    }

    // Producto de matrices
    template<typename T>
    Tensor<T,2> matrix_product(const Tensor<T,2>& A,
                               const Tensor<T,2>& B) {
        auto a = A.shape();
        auto b = B.shape();
        if (a[1] != b[0])
            throw std::runtime_error("matrix_product: inner dims mismatch");
        Tensor<T,2> out(a[0], b[1]);
        for (size_t i = 0; i < a[0]; ++i) {
            for (size_t j = 0; j < b[1]; ++j) {
                T sum = T(0);
                for (size_t k = 0; k < a[1]; ++k)
                    sum += A(i,k) * B(k,j);
                out(i,j) = sum;
            }
        }
        return out;
    }

}
#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H