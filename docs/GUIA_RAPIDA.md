# Guía Rápida de Uso - PONG AI Framework

## 🚀 Inicio Rápido

### 1. Crear una Red Neuronal Simple

```cpp
#include "include/utec/nn/neural_network.h"
#include "include/utec/nn/nn_dense.h"
#include "include/utec/nn/nn_activation.h"

using namespace utec::neural_network;
using namespace utec::algebra;

// Crear una red: 2 inputs -> 4 hidden -> 1 output
NeuralNetwork<float> network;
network.add_layer(std::make_unique<Dense<float>>(2, 4));
network.add_layer(std::make_unique<ReLU<float>>());
network.add_layer(std::make_unique<Dense<float>>(4, 1));
```

### 2. Preparar Datos

```cpp
// Datos XOR
Tensor<float, 2> X(4, 2);
X(0,0)=0; X(0,1)=0;  // 0 XOR 0 = 0
X(1,0)=0; X(1,1)=1;  // 0 XOR 1 = 1
X(2,0)=1; X(2,1)=0;  // 1 XOR 0 = 1
X(3,0)=1; X(3,1)=1;  // 1 XOR 1 = 0

Tensor<float, 2> Y(4, 1);
Y(0,0)=0; Y(1,0)=1; Y(2,0)=1; Y(3,0)=0;
```

### 3. Entrenar la Red

```cpp
// Entrenar con parámetros automáticos (MSE + SGD)
float loss = network.train(X, Y, 1000, 0.1f);
std::cout << "Loss final: " << loss << "\n";
```

### 4. Hacer Predicciones

```cpp
auto predictions = network.forward(X);
for (int i = 0; i < 4; ++i) {
    std::cout << "Entrada: [" << X(i,0) << "," << X(i,1)
              << "] -> Predicción: " << predictions(i,0) << "\n";
}
```

## 📚 Ejemplos Comunes

### Ejemplo 1: Clasificación Binaria con Early Stopping

```cpp
// Crear red más compleja
NeuralNetwork<float> net;
net.add_layer(std::make_unique<Dense<float>>(10, 32));
net.add_layer(std::make_unique<ReLU<float>>());
net.add_layer(std::make_unique<Dense<float>>(32, 16));
net.add_layer(std::make_unique<ReLU<float>>());
net.add_layer(std::make_unique<Dense<float>>(16, 1));

// Entrenar con early stopping
auto metrics = net.train_advanced(
    X_train, Y_train,
    500,        // max epochs
    0.01f,      // learning rate
    30,         // patience (parar si 30 épocas sin mejora)
    1e-6f       // min_delta
);

std::cout << "Épocas: " << metrics.epochs_trained << "\n";
std::cout << "Convergió: " << (metrics.converged ? "Sí" : "No") << "\n";
std::cout << "Mejor loss: " << metrics.best_loss << "\n";
```

### Ejemplo 2: Evaluación Completa

```cpp
// Dividir datos en entrenamiento y prueba
// ... supongamos X_train, Y_train, X_test, Y_test ...

// Entrenar
auto metrics = net.train_advanced(X_train, Y_train, 1000, 0.01f, 50, 1e-6f);

// Evaluar en datos de prueba
auto eval_metrics = net.evaluate(X_test, Y_test);

std::cout << "Test Loss: " << eval_metrics.test_loss << "\n";
std::cout << "Accuracy: " << (eval_metrics.accuracy * 100) << "%\n";
std::cout << "MAE: " << eval_metrics.mean_absolute_error << "\n";

// Hacer predicciones
auto predictions = net.forward(X_test);
```

### Ejemplo 3: Arquitecturas Diferentes

```cpp
// Arquitectura profunda (4 capas ocultas)
NeuralNetwork<float> deep_net;
deep_net.add_layer(std::make_unique<Dense<float>>(10, 128));
deep_net.add_layer(std::make_unique<ReLU<float>>());
deep_net.add_layer(std::make_unique<Dense<float>>(128, 64));
deep_net.add_layer(std::make_unique<ReLU<float>>());
deep_net.add_layer(std::make_unique<Dense<float>>(64, 32));
deep_net.add_layer(std::make_unique<ReLU<float>>());
deep_net.add_layer(std::make_unique<Dense<float>>(32, 1));

// Arquitectura simple (sin capas ocultas)
NeuralNetwork<float> simple_net;
simple_net.add_layer(std::make_unique<Dense<float>>(10, 1));

// Arquitectura con Sigmoid (para probabilidades)
NeuralNetwork<float> prob_net;
prob_net.add_layer(std::make_unique<Dense<float>>(20, 16));
prob_net.add_layer(std::make_unique<ReLU<float>>());
prob_net.add_layer(std::make_unique<Dense<float>>(16, 1));
prob_net.add_layer(std::make_unique<Sigmoid<float>>());
```

### Ejemplo 4: Agente Pong

```cpp
#include "include/utec/agent/PongAgent.h"

// Crear red neuronal para el agente
auto agent_net = std::make_unique<NeuralNetwork<float>>();
agent_net->add_layer(std::make_unique<Dense<float>>(3, 16));
agent_net->add_layer(std::make_unique<ReLU<float>>());
agent_net->add_layer(std::make_unique<Dense<float>>(16, 3));

// Crear agente
PongAgent<float> agent(std::move(agent_net));

// Crear ambiente
EnvGym env;

// Jugar episodio
State state = env.reset();
bool done = false;
int steps = 0;

while (!done && steps < 100) {
    int action = agent.act(state);  // -1 (arriba), 0 (quedo), 1 (abajo)
    float reward;
    state = env.step(action, reward, done);
    std::cout << "Step " << steps << ": action=" << action 
              << ", reward=" << reward << "\n";
    steps++;
}
```

## 🔧 Hiperparámetros Recomendados

### Para Problemas Simples (como XOR)
```cpp
float learning_rate = 0.1f;    // Más alto para converger rápido
size_t epochs = 1000;
size_t batch_size = 32;        // Automático (interno)
```

### Para Problemas Moderados
```cpp
float learning_rate = 0.01f;
size_t max_epochs = 500;
size_t patience = 30;
float min_delta = 1e-6f;
```

### Para Problemas Complejos
```cpp
float learning_rate = 0.001f;  // Más bajo para estabilidad
size_t max_epochs = 2000;
size_t patience = 50;          // Más paciente
float min_delta = 1e-7f;
```

## 📊 Operaciones Tensor

```cpp
// Crear tensores
Tensor<float, 2> A(3, 4);    // Matriz 3x4
Tensor<float, 1> v(5);       // Vector de 5 elementos

// Acceso
float value = A(1, 2);       // Elemento [1,2]
A(1, 2) = 3.14f;

// Operaciones
auto B = A * 2.0f;           // Multiplicación por escalar
auto C = A + B;              // Suma elemento-wise
auto D = A * B;              // Multiplicación elemento-wise

// Álgebra lineal
auto AT = transpose(A);      // Transposición
auto C = matrix_product(A, B);  // Multiplicación matricial

// Información
auto shape = A.shape();      // std::array<size_t, 2>
size_t elements = A.size();  // Número total de elementos

// Llenado
A.fill(0.0f);               // Llenar con un valor
```

## ⚡ Tips de Optimización

### 1. Normalización de Datos
```cpp
// Normalizar inputs a [0, 1]
float min_x = 0, max_x = 1;
for (size_t i = 0; i < X.shape()[0]; ++i) {
    for (size_t j = 0; j < X.shape()[1]; ++j) {
        X(i,j) = (X(i,j) - min_x) / (max_x - min_x);
    }
}
```

### 2. Monitoreo de Convergencia
```cpp
auto metrics = net.train_advanced(X, Y, 1000, 0.01f, 50, 1e-6f);

std::cout << "Loss history (cada 100 épocas):\n";
for (size_t i = 0; i < metrics.loss_history.size(); i += 100) {
    std::cout << "Época " << i << ": " << metrics.loss_history[i] << "\n";
}
```

### 3. Ajuste Fino de Learning Rate
```cpp
// Empezar con lr alto y reducir
for (float lr : {0.1f, 0.01f, 0.001f}) {
    std::cout << "Learning rate: " << lr << "\n";
    auto m = net.train_advanced(X, Y, 100, lr, 20, 1e-6f);
    std::cout << "Final loss: " << m.final_loss << "\n";
}
```

## 🐛 Debugging

### Ver Dimensiones
```cpp
auto shape = X.shape();
std::cout << "X shape: [" << shape[0] << ", " << shape[1] << "]\n";

// Verificar compatibilidad
if (X.shape()[0] != Y.shape()[0]) {
    std::cerr << "Error: X e Y tienen diferentes números de muestras\n";
}
```

### Verificar Predicciones
```cpp
auto pred = net.forward(X);
std::cout << "Predicción shape: [" << pred.shape()[0] 
          << ", " << pred.shape()[1] << "]\n";
std::cout << "Primera predicción: " << pred(0, 0) << "\n";
```

### Monitoreo de Pérdida
```cpp
auto metrics = net.train_advanced(X, Y, 50, 0.01f, 100, 1e-10f);

if (metrics.converged) {
    std::cout << "Red convergió en " << metrics.epochs_trained << " épocas\n";
} else {
    std::cout << "Red NO convergió (probablemente necesita más épocas)\n";
}

std::cout << "Pérdida inicial: " << metrics.loss_history.front() << "\n";
std::cout << "Pérdida final: " << metrics.loss_history.back() << "\n";
```

## 🎓 Recursos Adicionales

- `docs/ARQUITECTURA.md` - Detalles de la arquitectura
- `examples/train_xor.cpp` - Ejemplo completo XOR
- `examples/train_pong_agent.cpp` - Ejemplo agente Pong
- `tests/` - Suite de pruebas unitarias

## 📞 Troubleshooting Común

| Problema | Solución |
|----------|----------|
| Red no converge | Reducir learning rate, aumentar épocas |
| NaN en pérdida | Verificar normalización de datos |
| Acceso inválido | Verificar dimensiones con `.shape()` |
| Compilación falla | Verificar includes correctos: `nn_*.h` |
| Memoria agotada | Reducir tamaño de red o datos |


