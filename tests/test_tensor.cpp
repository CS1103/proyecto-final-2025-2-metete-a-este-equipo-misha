#include "../include/utec/algebra/tensor.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <random>
#include <iomanip>

using namespace utec::algebra;

class TensorTestSuite {
private:
    size_t tests_passed = 0;
    size_t tests_total = 0;

    void assert_test(bool condition, const std::string& test_name) {
        tests_total++;
        if (condition) {
            tests_passed++;
            std::cout << "✓ " << test_name << " PASSED\n";
        } else {
            std::cout << "✗ " << test_name << " FAILED\n";
        }
    }

public:
    void run_all_tests() {
        std::cout << "=== COMPREHENSIVE TENSOR TEST SUITE ===\n\n";

        // Original test cases
        test_case_1();
        test_case_2();
        test_case_3();
        test_case_4();
        test_case_5();
        test_case_6();
        test_case_7();

        // Enhanced test cases
        test_edge_cases();
        test_performance_scalability();
        test_memory_safety();
        test_numerical_stability();
        test_broadcasting_advanced();
        test_error_handling_comprehensive();

        print_summary();
    }

private:
    void test_case_1() {
        std::cout << "Test Case 1: Creación, acceso y fill\n";
        Tensor<int,2> t(2,3);
        t.fill(7);
        int x = t(1,2);
        assert_test(x == 7, "Basic creation and fill");
        std::cout << "\n";
    }

    void test_case_2() {
        std::cout << "Test Case 2: Reshape válido y acceso lineal\n";
        Tensor<int,2> t2(2,3);
        for (int i = 0; i < 6; ++i) {
            t2[i] = i + 1;
        }
        t2.reshape({3,2});
        int y = t2[5];
        int z = t2(2,1);
        assert_test(y == z && y == 6, "Reshape and linear access");
        std::cout << "\n";
    }

    void test_case_3() {
        std::cout << "Test Case 3: Reshape inválido\n";
        Tensor<int,3> t3(2,2,2);
        try {
            t3.reshape({2,4,1});
            assert_test(false, "Invalid reshape should throw");
        } catch (const std::invalid_argument&) {
            assert_test(true, "Exception handling for invalid reshape");
        }
        std::cout << "\n";
    }

    void test_case_4() {
        std::cout << "Test Case 4: Suma y resta de tensores\n";
        Tensor<double,2> a(2,2), b(2,2);
        a.fill(0.0);
        a(0,1) = 5.5;
        b.fill(2.0);
        auto sum = a + b;
        auto diff = sum - b;

        bool sum_correct = std::abs(sum(0,1) - 7.5) < 1e-9;
        bool diff_correct = std::abs(diff(0,1) - 5.5) < 1e-9;
        assert_test(sum_correct && diff_correct, "Tensor arithmetic operations");
        std::cout << "\n";
    }

    void test_case_5() {
        std::cout << "Test Case 5: Multiplicación escalar y tensores 3D\n";
        Tensor<float,1> v(3);
        v.fill(2.0f);
        auto scaled = v * 4.0f;

        Tensor<int,3> cube(2,2,2);
        cube.fill(1);
        auto cube2 = cube * cube;

        bool scalar_correct = std::abs(scaled(2) - 8.0f) < 1e-6f;
        bool cube_correct = cube2(1,1,1) == 1;
        assert_test(scalar_correct && cube_correct, "Scalar multiplication and 3D tensors");
        std::cout << "\n";
    }

    void test_case_6() {
        std::cout << "Test Case 6: Broadcasting implícito\n";
        Tensor<int,2> m(2,1);
        m(0,0) = 3; m(1,0) = 4;
        Tensor<int,2> n(2,3);
        n.fill(5);
        auto p = m * n;

        bool broadcast_correct = (p(0,2) == 15) && (p(1,1) == 20);
        assert_test(broadcast_correct, "Broadcasting multiplication");
        std::cout << "\n";
    }

    void test_case_7() {
        std::cout << "Test Case 7: Transpose 2D\n";
        Tensor<int,2> m2(2,3);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                m2(i,j) = i * 3 + j;
            }
        }
        auto mt = m2.transpose_2d();

        bool shape_correct = (mt.shape()[0] == 3) && (mt.shape()[1] == 2);
        bool data_correct = (mt(0,1) == m2(1,0)) && (mt(2,0) == m2(0,2));
        assert_test(shape_correct && data_correct, "2D transpose operation");
        std::cout << "\n";
    }

    void test_edge_cases() {
        std::cout << "Advanced Test: Edge Cases\n";

        // Test with very small tensors
        Tensor<double,1> tiny(1);
        tiny(0) = 42.0;
        auto tiny_scaled = tiny * 2.0;
        assert_test(tiny_scaled(0) == 84.0, "Single element tensor");

        // Test with zero-filled tensors
        Tensor<float,2> zeros(5,5);
        zeros.fill(0.0f);
        auto zeros_sum = zeros + zeros;
        bool all_zeros = true;
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                if (zeros_sum(i,j) != 0.0f) all_zeros = false;
            }
        }
        assert_test(all_zeros, "Zero tensor operations");

        // Test with negative values
        Tensor<int,2> negative(3,3);
        negative.fill(-5);
        auto abs_result = negative * negative;
        assert_test(abs_result(1,1) == 25, "Negative value operations");

        std::cout << "\n";
    }

    void test_performance_scalability() {
        std::cout << "Advanced Test: Performance and Scalability\n";

        // Test with progressively larger tensors
        std::vector<size_t> sizes = {10, 50, 100, 200};
        std::vector<double> times;

        for (auto size : sizes) {
            auto start = std::chrono::high_resolution_clock::now();

            Tensor<float,2> large_a(size, size);
            Tensor<float,2> large_b(size, size);
            large_a.fill(1.5f);
            large_b.fill(2.0f);

            // Perform multiple operations to test scalability
            auto result1 = large_a + large_b;
            auto result2 = result1 * large_a;
            auto result3 = result2.transpose_2d();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }

        // Check that performance scales reasonably (not exponentially worse)
        bool scalable = true;
        for (size_t i = 1; i < times.size(); ++i) {
            double ratio = times[i] / times[i-1];
            double size_ratio = static_cast<double>(sizes[i] * sizes[i]) / (sizes[i-1] * sizes[i-1]);
            if (ratio > size_ratio * 10) { // Allow some overhead, but not 10x worse than O(n²)
                scalable = false;
                break;
            }
        }

        assert_test(scalable, "Performance scales reasonably with tensor size");
        std::cout << "\n";
    }

    void test_memory_safety() {
        std::cout << "Advanced Test: Memory Safety\n";

        // Test that temporary objects are handled correctly
        {
            Tensor<double,2> temp1(100, 100);
            temp1.fill(1.0);
            {
                Tensor<double,2> temp2(100, 100);
                temp2.fill(2.0);
                auto result = temp1 + temp2;
                assert_test(result(50,50) == 3.0, "Temporary object handling");
            }
            // temp2 should be destroyed here, but result should still be valid
        }

        // Test copy semantics
        Tensor<int,2> original(5,5);
        original.fill(10);
        auto copy = original;
        copy.fill(20);
        assert_test(original(0,0) == 10 && copy(0,0) == 20, "Copy semantics work correctly");

        std::cout << "\n";
    }

    void test_numerical_stability() {
        std::cout << "Advanced Test: Numerical Stability\n";

        // Test with very small numbers
        Tensor<double,2> small_nums(3,3);
        small_nums.fill(1e-10);
        auto small_result = small_nums + small_nums;
        assert_test(std::abs(small_result(1,1) - 2e-10) < 1e-15, "Small number precision");

        // Test with very large numbers
        Tensor<double,2> large_nums(3,3);
        large_nums.fill(1e10);
        auto large_result = large_nums * 2.0;
        assert_test(std::abs(large_result(1,1) - 2e10) < 1e5, "Large number handling");

        std::cout << "\n";
    }

    void test_broadcasting_advanced() {
        std::cout << "Advanced Test: Advanced Broadcasting\n";

        // Test multiple broadcasting scenarios
        Tensor<float,2> a(3,1);
        Tensor<float,2> b(1,4);
        a.fill(2.0f);
        b.fill(3.0f);

        auto broadcasted = a * b;
        bool correct_shape = (broadcasted.shape()[0] == 3) && (broadcasted.shape()[1] == 4);
        bool correct_values = (broadcasted(0,0) == 6.0f) && (broadcasted(2,3) == 6.0f);
        assert_test(correct_shape && correct_values, "Complex broadcasting scenarios");

        std::cout << "\n";
    }

    void test_error_handling_comprehensive() {
        std::cout << "Advanced Test: Comprehensive Error Handling\n";

        size_t exceptions_caught = 0;

        // Test out-of-bounds access
        try {
            Tensor<int,2> bounds_test(5,5);
            bounds_test(10,10); // Should throw
        } catch (const std::exception&) {
            exceptions_caught++;
        }

        // Test shape mismatch in operations
        try {
            Tensor<int,2> a(2,3);
            Tensor<int,2> b(3,2);
            auto result = a + b; // Should throw
        } catch (const std::exception&) {
            exceptions_caught++;
        }

        // Test invalid linear access
        try {
            Tensor<int,1> linear_test(10);
            linear_test[15]; // Should throw
        } catch (const std::exception&) {
            exceptions_caught++;
        }

        assert_test(exceptions_caught >= 2, "Proper exception handling for invalid operations");
        std::cout << "\n";
    }

    void print_summary() {
        std::cout << "\n=== TEST SUMMARY ===\n";
        std::cout << "Tests passed: " << tests_passed << "/" << tests_total;
        double percentage = (static_cast<double>(tests_passed) / tests_total) * 100.0;
        std::cout << " (" << std::fixed << std::setprecision(1) << percentage << "%)\n";

        if (tests_passed == tests_total) {
            std::cout << "🎉 ALL TESTS PASSED! Excellent implementation.\n";
        } else {
            std::cout << "⚠️  Some tests failed. Please review implementation.\n";
        }
        std::cout << "====================\n";
    }
};

int main() {
    TensorTestSuite suite;
    suite.run_all_tests();
    return 0;
}
