cmake_minimum_required(VERSION 3.31)
project(PONG_AI VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard with strict requirements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and debugging flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined")

# Find OpenMP for potential parallelization
find_package(OpenMP)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/**/*.cpp")

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
endif()

# Main executable
add_executable(PONG_AI main.cpp ${SOURCES})

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(PONG_AI OpenMP::OpenMP_CXX)
    target_compile_definitions(PONG_AI PRIVATE UTEC_OPENMP_ENABLED)
endif()

# Example executables
add_executable(train_xor examples/train_xor.cpp ${SOURCES})
add_executable(train_pong_agent examples/train_pong_agent.cpp ${SOURCES})

# Link OpenMP to examples if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(train_xor OpenMP::OpenMP_CXX)
    target_link_libraries(train_pong_agent OpenMP::OpenMP_CXX)
endif()

# === TESTING FRAMEWORK ===
enable_testing()

# Test executables with proper linking
add_executable(test_tensor tests/test_tensor.cpp ${SOURCES})
add_executable(test_neural_network tests/test_neural_network.cpp ${SOURCES})
add_executable(test_agent_env tests/test_agent_env.cpp ${SOURCES})

# Performance benchmark executable
add_executable(performance_benchmark benchmarks/performance_tests.cpp ${SOURCES})

# Link OpenMP to performance tests for parallel benchmarks
if(OpenMP_CXX_FOUND)
    target_link_libraries(performance_benchmark OpenMP::OpenMP_CXX)
    target_compile_definitions(performance_benchmark PRIVATE UTEC_OPENMP_ENABLED)
endif()

# Register tests with CTest
add_test(NAME TensorOperations COMMAND test_tensor)
add_test(NAME NeuralNetwork COMMAND test_neural_network)
add_test(NAME AgentEnvironment COMMAND test_agent_env)
add_test(NAME PerformanceBenchmark COMMAND performance_benchmark)

# Set test properties
set_tests_properties(TensorOperations PROPERTIES TIMEOUT 30)
set_tests_properties(NeuralNetwork PROPERTIES TIMEOUT 60)
set_tests_properties(AgentEnvironment PROPERTIES TIMEOUT 45)
set_tests_properties(PerformanceBenchmark PROPERTIES TIMEOUT 120)

# === INSTALLATION ===
install(TARGETS PONG_AI DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES README.md docs/BIBLIOGRAFIA.md DESTINATION share/doc/pong_ai)

# === DOCUMENTATION ===
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
                   ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc ALL
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif()

# === PACKAGING ===
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Pong AI - Neural Network Implementation in C++20")
set(CPACK_PACKAGE_VENDOR "UTEC CS2013")
include(CPack)

# === CUSTOM TARGETS ===
# Run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_tensor test_neural_network test_agent_env
    COMMENT "Running all unit tests")

# Run performance benchmarks
add_custom_target(benchmark
    COMMAND performance_benchmark
    DEPENDS performance_benchmark
    COMMENT "Running performance benchmarks")

# Code coverage (requires gcov)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage")

    add_custom_target(coverage
        COMMAND gcov -r ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating code coverage report")
endif()

# Print configuration summary
message(STATUS "=== PONG AI Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "OpenMP Support: ${OpenMP_CXX_FOUND}")
message(STATUS "Doxygen Found: ${DOXYGEN_FOUND}")
message(STATUS "=====================================")

# Header-only library check (ensures no missing implementations)
set_target_properties(PONG_AI PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
