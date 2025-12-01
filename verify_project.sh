#!/bin/bash
# Script de verificación rápida del proyecto

echo "======================================="
echo "VERIFICACIÓN DEL PROYECTO PONG AI"
echo "======================================="
echo ""

echo "1. Verificando estructura de directorios..."
for dir in include/utec/{algebra,nn,agent} src tests examples docs; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir existe"
    else
        echo "   ✗ FALTA: $dir"
    fi
done
echo ""

echo "2. Verificando archivos header principales..."
for file in include/utec/algebra/tensor.h \
            include/utec/nn/neural_network.h \
            include/utec/nn/nn_dense.h \
            include/utec/nn/nn_activation.h \
            include/utec/nn/nn_loss.h \
            include/utec/nn/nn_optimizer.h \
            include/utec/nn/nn_interfaces.h \
            include/utec/agent/PongAgent.h; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ FALTA: $file"
    fi
done
echo ""

echo "3. Verificando archivos de documentación..."
for file in docs/ARQUITECTURA.md docs/GUIA_RAPIDA.md docs/CAMBIOS_REALIZADOS.md README.md; do
    if [ -f "$file" ]; then
        wc=$(wc -l < "$file")
        echo "   ✓ $file ($wc líneas)"
    else
        echo "   ✗ FALTA: $file"
    fi
done
echo ""

echo "4. Verificando archivos de prueba..."
for file in tests/test_tensor.cpp tests/test_neural_network.cpp tests/test_agent_env.cpp; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ FALTA: $file"
    fi
done
echo ""

echo "5. Verificando ejemplos..."
for file in examples/train_xor.cpp examples/train_pong_agent.cpp; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ FALTA: $file"
    fi
done
echo ""

echo "6. Verificando includes en archivos..."
echo "   Buscando includes incorrectos..."

# Buscar includes que apunten a archivos sin prefijo nn_
if grep -r '#include.*activation.h' include/utec/nn/ 2>/dev/null | grep -v nn_activation; then
    echo "   ✗ Encontrado include incorrecto de activation.h"
else
    echo "   ✓ Sin includes incorrectos de activation.h"
fi

if grep -r '#include.*dense.h' include/utec/nn/ 2>/dev/null | grep -v nn_dense; then
    echo "   ✗ Encontrado include incorrecto de dense.h"
else
    echo "   ✓ Sin includes incorrectos de dense.h"
fi

echo ""
echo "======================================="
echo "VERIFICACIÓN COMPLETADA"
echo "======================================="

