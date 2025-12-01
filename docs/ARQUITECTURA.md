# Arquitectura del Proyecto PONG AI - Epic 3

## 📋 Descripción General

Este documento detalla la arquitectura y diseño del framework de redes neuronales implementado en C++20 para el proyecto PONG AI, focusing en el Epic 3 (Aplicación y Documentación).

## 🏗️ Estructura de Carpetas

```
include/utec/
├── algebra/                   # Álgebra Lineal (Epic 1 - No modificar)
│   └── tensor.h              # Tensor<T, Rank> - Arrays multidimensionales
├── nn/                        # Red Neuronal (Epic 2 - No modificar)
│   ├── nn_interfaces.h       # Interfaces: ILayer<T>, IOptimizer<T>, ILoss<T>
│   ├── nn_dense.h            # Capas densas (fully connected)
│   ├── nn_activation.h       # Funciones de activación (ReLU, Sigmoid)
│   ├── nn_loss.h             # Funciones de pérdida (MSE, BCE)
│   ├── nn_optimizer.h        # Optimizadores (SGD, Adam)
│   └── neural_network.h      # Clase principal de red neuronal
└── agent/                     # Agentes y Aplicaciones (Epic 3)
    └── PongAgent.h           # Agente de Pong + Ambiente de simulación
```

## 🔑 Componentes Principales

### 1. **Tensor<T, Rank>** (Epic 1)
- **Ubicación**: `include/utec/algebra/tensor.h`
- **Descripción**: Arrays multidimensionales genéricos en C++20
- **Características**:
  - Acceso variádico: `tensor(i, j, k, ...)`
  - Broadcasting automático para operaciones
  - Operaciones de matriz: transposición, multiplicación
  - Iteradores eficientes para recorrido secuencial

**Métodos clave**:
```cpp
Tensor<T, N> tensor(d1, d2, ..., dN);  // Constructor
T& operator()(idx...);                  // Acceso
std::array<size_t, N> shape();         // Dimensiones
Tensor<T,2> matrix_product(A, B);      // Multiplicación matricial
Tensor<T,2> transpose(M);               // Transposición
```

### 2. **Interfaz de Capas** (Epic 2)

#### ILayer<T>
- **Ubicación**: `include/utec/nn/nn_interfaces.h`
- **Métodos virtuales**:
  ```cpp
  virtual Tensor<T,2> forward(const Tensor<T,2>& x) = 0;
  virtual Tensor<T,2> backward(const Tensor<T,2>& gradient) = 0;
  virtual void update_params(IOptimizer<T>& optimizer) {}
  ```

#### Capas Implementadas

**Dense (Fully Connected)**
- **Ubicación**: `include/utec/nn/nn_dense.h`
- **Parámetros**: pesos (W) y sesgos (b)
- **Forward**: `Z = X·W + b`
- **Backward**: Calcula gradientes ∇W, ∇b, ∇X
- **Inicialización**: Xavier por defecto

```cpp
Dense<float> layer(input_dim, output_dim);
auto output = layer.forward(input);
auto grad_input = layer.backward(grad_output);
```

**ReLU (Rectified Linear Unit)**
- **Ubicación**: `include/utec/nn/nn_activation.h`
- **Forward**: `y = max(0, x)`
- **Backward**: `dy/dx = 1 si x > 0, else 0`

**Sigmoid**
- **Forward**: `y = 1 / (1 + e^(-x))`
- **Backward**: `dy/dx = σ(x) * (1 - σ(x))`

### 3. **Funciones de Pérdida** (Epic 2)

#### MSELoss (Mean Squared Error)
- **Ubicación**: `include/utec/nn/nn_loss.h`
- **Fórmula**: `L = (1/N) * Σ(ŷ - y)²`
- **Gradiente**: `dL/dŷ = 2(ŷ - y)/N`
- **Uso**: Problemas de regresión

```cpp
MSELoss<float> loss(predictions, targets);
float loss_value = loss.loss();
auto gradient = loss.loss_gradient();
```

#### BCELoss (Binary Cross Entropy)
- **Fórmula**: `L = -(1/N) * Σ[y*log(p) + (1-y)*log(1-p)]`
- **Gradiente**: `dL/dŷ = (p - y)/(p*(1-p)*N)`
- **Uso**: Clasificación binaria

### 4. **Optimizadores** (Epic 2)

#### SGD (Stochastic Gradient Descent)
- **Ubicación**: `include/utec/nn/nn_optimizer.h`
- **Actualización**: `θ = θ - α∇L`
- **Parámetros**: learning_rate (α)

```cpp
SGD<float> optimizer(0.01f);
optimizer.update(weights, gradients);
```

#### Adam (Adaptive Moment Estimation)
- **Características**: Momentos adaptivos de primer y segundo orden
- **Parámetros**: 
  - `learning_rate` (típicamente 0.001)
  - `beta1` = 0.9 (decaimiento del primer momento)
  - `beta2` = 0.999 (decaimiento del segundo momento)
  - `epsilon` = 1e-8 (estabilidad numérica)

```cpp
Adam<float> optimizer(0.001f, 0.9f, 0.999f, 1e-8f);
optimizer.update(weights, gradients);
optimizer.step();
```

### 5. **Red Neuronal** (Epic 2/3)

#### NeuralNetwork<T>
- **Ubicación**: `include/utec/nn/neural_network.h`
- **Características**:
  - Composición flexible de capas
  - Entrenamiento con mini-batches
  - Early stopping
  - Métricas de evaluación

**Métodos principales**:

```cpp
NeuralNetwork<float> net;

// Agregar capas
net.add_layer(std::make_unique<Dense<float>>(2, 4));
net.add_layer(std::make_unique<ReLU<float>>());

// Forward pass (predicción)
auto output = net.forward(input);

// Entrenamiento básico (MSE + SGD por defecto)
float final_loss = net.train(X, Y, epochs, learning_rate);

// Entrenamiento avanzado (con early stopping)
auto metrics = net.train_advanced(
    X, Y,                    // Datos de entrada y salida
    max_epochs,              // Número máximo de épocas
    learning_rate,           // Tasa de aprendizaje
    patience,                // Épocas sin mejora antes de parar
    min_delta                // Mejora mínima considerada como progreso
);

// Evaluación en datos de prueba
auto eval = net.evaluate(X_test, Y_test);
// Retorna: test_loss, accuracy, mean_absolute_error
```

**Estructuras de Métricas**:

```cpp
// TrainingMetrics
struct TrainingMetrics<T> {
    size_t epochs_trained;              // Épocas ejecutadas
    T final_loss;                       // Pérdida final
    T best_loss;                        // Mejor pérdida alcanzada
    bool converged;                     // ¿Convergió tempranamente?
    std::vector<T> loss_history;        // Histórico de pérdida por época
};

// EvaluationMetrics
struct EvaluationMetrics<T> {
    T test_loss;                        // MSE en datos de prueba
    T accuracy;                         // Precisión (0-1)
    T mean_absolute_error;              // Error absoluto promedio
};
```

### 6. **Agente de Pong** (Epic 3)

#### PongAgent<T>
- **Ubicación**: `include/utec/agent/PongAgent.h`
- **Descripción**: Agente que aprende a jugar Pong usando una red neuronal

```cpp
struct State {
    float ball_x, ball_y;     // Posición de la bola
    float paddle_y;            // Posición de la paleta
};

PongAgent<float> agent(neural_network);
int action = agent.act(state);  // Retorna: -1 (arriba), 0 (quedo), 1 (abajo)
```

#### EnvGym (Simulador de Pong)
- **Métodos**:
  ```cpp
  State reset();                              // Reinicia el juego
  State step(action, reward, done);           // Ejecuta una acción
  ```
- **Recompensas**:
  - `+1.0` cuando golpea la bola exitosamente
  - `+0.5` cuando la bola llega al lado del oponente
  - `-1.0` cuando falla y pierde

## 📊 Flujo de Entrenamiento

```
┌─────────────────────────────────────────────────────┐
│ 1. PREPARACIÓN DE DATOS                             │
│   - Crear tensores X (entrada) e Y (salida)        │
│   - Normalizar/escalar si es necesario              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ 2. DEFINICIÓN DE ARQUITECTURA                       │
│   - add_layer(Dense)                                │
│   - add_layer(ReLU/Sigmoid)                         │
│   - Repetir según sea necesario                     │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ 3. ENTRENAMIENTO (por cada época)                   │
│   ┌─────────────────────────────────────────┐      │
│   │ Para cada mini-batch:                   │      │
│   │  a) Forward pass: ŷ = net(x)            │      │
│   │  b) Calcular pérdida: L = loss(ŷ, y)   │      │
│   │  c) Backward pass: ∇L                   │      │
│   │  d) Actualizar parámetros: θ -= α∇L    │      │
│   └─────────────────────────────────────────┘      │
│                                                     │
│   Si: early_stopping activado                      │
│   ├─ Evaluar en validación                         │
│   ├─ Si no mejora en 'patience' épocas → STOP      │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│ 4. EVALUACIÓN Y PREDICCIÓN                          │
│   - evaluate(X_test, Y_test)                        │
│   - forward(X_new)                                  │
└─────────────────────────────────────────────────────┘
```

## 🔄 Algoritmos Implementados

### Forward Propagation
Para cada capa i:
```
a[i] = σ(z[i])  donde z[i] = a[i-1] · W[i] + b[i]
```

### Backward Propagation
Para cada capa i (de atrás hacia adelante):
```
dz[i] = σ'(z[i]) * da[i]
dW[i] = (1/m) * a[i-1]ᵀ · dz[i]
db[i] = (1/m) * Σ dz[i]  (suma sobre muestras)
da[i-1] = dz[i] · W[i]ᵀ
```

### Actualización de Parámetros (SGD)
```
W = W - α * dW
b = b - α * db
```

### Actualización de Parámetros (Adam)
```
m_t = β1 * m_{t-1} + (1-β1) * g_t
v_t = β2 * v_{t-1} + (1-β2) * g_t²
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)
θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
```

## 📈 Complejidad Computacional

| Operación | Complejidad | Descripción |
|-----------|------------|------------|
| Forward (Dense) | O(n·m) | n inputs, m outputs |
| Backward (Dense) | O(n·m) | Gradientes de pesos |
| Matrix Product | O(n·m·k) | Multiplicación matricial NxM por MxK |
| MSELoss | O(n) | n predicciones |
| Adam Update | O(n) | n parámetros |

## 🧪 Pruebas

### Test Files
- `tests/test_tensor.cpp` - Pruebas de operaciones Tensor
- `tests/test_neural_network.cpp` - Pruebas de capas y funciones de pérdida
- `tests/test_agent_env.cpp` - Pruebas del agente Pong

### Ejemplos
- `examples/train_xor.cpp` - Entrenamiento en problema XOR
- `examples/train_pong_agent.cpp` - Entrenamiento del agente Pong

## 🚀 Compilación y Ejecución

```bash
# Compilar
cmake --build cmake-build-debug

# Ejecutar programa principal
./PONG_AI

# Ejecutar ejemplo XOR
./train_xor

# Ejecutar tests
./test_tensor
./test_neural_network
./test_agent_env
```

## 📝 Consideraciones de Diseño

### 1. **Genéricos (Templates)**
- Todo es templado en tipo `T` (float, double, etc.)
- Permite flexibilidad en precisión numérica

### 2. **Polimorfismo Virtual**
- ILayer<T>, IOptimizer<T>, ILoss<T> como interfaces
- Permite agregar nuevas capas/optimizadores fácilmente

### 3. **Smart Pointers**
- `std::unique_ptr` para gestión automática de memoria
- Evita memory leaks

### 4. **Move Semantics**
- `add_layer(std::make_unique<...>())` transfiere propiedad
- Eficiente sin copias innecesarias

### 5. **Broadcasting**
- Operaciones automáticas Tensor+Tensor y Tensor+Escalar
- Simplifica código de capas

## 🎯 Limitaciones y Futuras Mejoras

### Actuales
- Operaciones 2D principalmente (tablas de datos)
- Sin paralelización explícita (OpenMP preparado)
- Mini-batch size fijo (32)

### Futuras
- [ ] Operaciones SIMD vectorizadas
- [ ] GPU support (CUDA)
- [ ] Dropout, BatchNorm
- [ ] Modelos pre-entrenados
- [ ] Serialización (guardar/cargar modelos)

## 📚 Referencias Bibliográficas

Ver `docs/BIBLIOGRAFIA.md` para detalles de fuentes académicas.

