# Bibliografia - Pong AI Project

## Referencias Técnicas Principales

### 1. Deep Reinforcement Learning
- **Mnih, V., et al. (2015)**. "Human-level control through deep reinforcement learning". *Nature*, 518(7540), 529-533.
  - Base teórica para el uso de redes neuronales profundas en juegos
  - Algoritmo DQN (Deep Q-Network) aplicable a Pong
  - Técnicas de experience replay implementadas en nuestro agente

### 2. Neural Network Optimization
- **Kingma, D. P., & Ba, J. (2014)**. "Adam: A method for stochastic optimization". *arXiv preprint arXiv:1412.6980*.
  - Optimizador Adam para entrenamiento eficiente
  - Adaptive learning rates implementados en nuestro framework
  - Análisis de convergencia para redes neuronales

### 3. Tensor Operations and Linear Algebra
- **Golub, G. H., & Van Loan, C. F. (2013)**. "Matrix computations" (4th ed.). Johns Hopkins University Press.
  - Fundamentos matemáticos para operaciones tensoriales
  - Algoritmos eficientes para broadcasting y transposición
  - Análisis de complejidad computacional

### 4. C++ Template Metaprogramming
- **Vandevoorde, D., Josuttis, N. M., & Gregor, D. (2017)**. "C++ Templates: The Complete Guide" (2nd ed.). Addison-Wesley.
  - Diseño de templates genéricos para Tensor<T, Rank>
  - Técnicas de optimización en tiempo de compilación
  - SFINAE y concepts para type safety

## Recursos Adicionales

### Algoritmos de Machine Learning
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement learning: An introduction" (2nd ed.). MIT Press.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep learning". MIT Press.

### Optimización y Performance
- Agner Fog. "Optimizing software in C++". Technical University of Denmark.
- Intel Corporation. "Intel® 64 and IA-32 Architectures Optimization Reference Manual".

### Software Engineering Best Practices
- Martin, R. C. (2017). "Clean Architecture: A Craftsman's Guide to Software Structure and Design". Prentice Hall.
- Meyers, S. (2014). "Effective Modern C++: 42 Specific Ways to Improve Your Use of C++11 and C++14". O'Reilly Media.

## Papers de Referencia para Pong AI

1. **Atari Domain**: Bellemare, M. G., et al. (2013). "The arcade learning environment: An evaluation platform for general agents". *JAIR*, 47, 253-279.

2. **Policy Gradient Methods**: Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning". *Machine learning*, 8(3-4), 229-256.

3. **Function Approximation**: Tsitsiklis, J. N., & Van Roy, B. (1997). "An analysis of temporal-difference learning with function approximation". *IEEE transactions on automatic control*, 42(5), 674-690.

4. **Neural Network Backpropagation**: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors". *Nature*, 323(6088), 533-536.

## Herramientas y Frameworks Consultados

- **Eigen**: C++ template library for linear algebra
- **PyTorch**: Para comparación de arquitecturas de red neuronal
- **OpenAI Gym**: Inspiración para el diseño de EnvGym
- **Google Test**: Framework de testing adoptado para nuestras pruebas

## Contribuciones Originales del Proyecto

- Implementación de broadcasting tensorial optimizada para C++20
- Framework de red neuronal header-only para máximo performance
- Agente Pong con arquitectura modular y extensible
- Sistema de testing comprehensivo con benchmarks de performance
