#!/bin/bash

# Simple compilation script for PONG AI without CMake
# This script compiles your neural network training examples using g++

echo "🚀 PONG AI - Simple Compilation Script"
echo "====================================="

# Check if g++ is available
if ! command -v g++ &> /dev/null; then
    echo "❌ Error: g++ compiler not found!"
    echo "Please install Xcode Command Line Tools:"
    echo "xcode-select --install"
    exit 1
fi

# Compilation flags
CXX_FLAGS="-std=c++20 -O2 -Wall -Wextra -I./include"

echo "📁 Working directory: $(pwd)"
echo "🔧 Compiler flags: $CXX_FLAGS"
echo ""

# Function to compile a single example
compile_example() {
    local example_name="$1"
    local source_file="examples/${example_name}.cpp"
    local output_file="${example_name}"

    if [ ! -f "$source_file" ]; then
        echo "❌ Source file not found: $source_file"
        return 1
    fi

    echo "🔨 Compiling $example_name..."
    if g++ $CXX_FLAGS "$source_file" -o "$output_file"; then
        echo "✅ Successfully compiled: $output_file"
        return 0
    else
        echo "❌ Compilation failed for $example_name"
        return 1
    fi
}

# Function to run an example
run_example() {
    local example_name="$1"
    local executable="./${example_name}"

    if [ ! -f "$executable" ]; then
        echo "❌ Executable not found: $executable"
        return 1
    fi

    echo ""
    echo "🎯 Running $example_name..."
    echo "================================"
    $executable
    echo "================================"
    echo "✅ $example_name completed!"
}

# Main compilation and execution
main() {
    echo "🔨 Starting compilation process..."
    echo ""

    # Compile main program
    echo "🔨 Compiling main program..."
    if g++ $CXX_FLAGS main.cpp -o main_demo; then
        echo "✅ Successfully compiled: main_demo"
    else
        echo "❌ Main compilation failed"
    fi

    # Compile ONLY VALID examples (those aligned with project)
    local examples=("train_xor" "train_pong_agent")
    local compiled_examples=()

    echo "📝 Compiling project-aligned examples:"
    echo "  ✅ train_xor.cpp - Validación de red neuronal"
    echo "  ✅ train_pong_agent.cpp - Entrenamiento del agente Pong (PRINCIPAL)"
    echo ""

    for example in "${examples[@]}"; do
        if compile_example "$example"; then
            compiled_examples+=("$example")
        fi
    done

    echo ""
    echo "📊 Compilation Summary:"
    echo "======================"
    echo "✅ Successfully compiled: ${#compiled_examples[@]} examples"

    if [ ${#compiled_examples[@]} -eq 0 ]; then
        echo "❌ No examples compiled successfully"
        return 1
    fi

    echo ""
    echo "🎮 Available programs:"
    echo "- ./main_demo          (Main framework demonstration)"
    for example in "${compiled_examples[@]}"; do
        if [ "$example" = "train_pong_agent" ]; then
            echo "- ./$example     (⭐ MAIN EXAMPLE - Pong agent training)"
        else
            echo "- ./$example           (${example//_/ } training)"
        fi
    done

    echo ""
    echo "🚀 Quick start - Run Pong agent training (MAIN EXAMPLE):"
    echo "./train_pong_agent"
    echo ""

    # Ask user what to run
    echo "Would you like to run a training example now? (y/n)"
    read -r response

    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Which example would you like to run?"
        echo "1) train_xor (Validating neural network architecture)"
        echo "2) train_pong_agent (⭐ MAIN - Training Pong agent)"
        echo "3) main_demo (Complete framework demonstration)"
        echo -n "Enter choice (1-3): "
        read -r choice

        case $choice in
            1)
                if [[ " ${compiled_examples[@]} " =~ " train_xor " ]]; then
                    run_example "train_xor"
                else
                    echo "❌ train_xor not available"
                fi
                ;;
            2)
                if [[ " ${compiled_examples[@]} " =~ " train_pong_agent " ]]; then
                    run_example "train_pong_agent"
                else
                    echo "❌ train_pong_agent not available"
                fi
                ;;
            3)
                if [ -f "./main_demo" ]; then
                    run_example "main_demo"
                else
                    echo "❌ main_demo not available"
                fi
                ;;
            *)
                echo "Invalid choice. You can run examples manually later."
                ;;
        esac
    fi

    echo ""
    echo "🎉 Setup complete! Your PONG AI neural network is ready."
    echo ""
    echo "💡 Project Focus:"
    echo "- This project trains a neural network to play PONG"
    echo "- train_pong_agent.cpp is the PRIMARY example"
    echo "- train_xor.cpp validates the architecture"
    echo ""
    echo "📚 Documentation:"
    echo "- See README.md for complete documentation"
    echo "- See docs/ANALISIS_EJEMPLOS.md for example analysis"
}

# Run main function
main "$@"
