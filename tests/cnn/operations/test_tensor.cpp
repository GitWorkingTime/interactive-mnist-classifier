#include "tensor.h"
#include <cassert>
#include <iostream>

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool floatEq(float a, float b, float tolerance = 0.001f) {
    return std::abs(a - b) < tolerance;
}

// ─── Constructor ──────────────────────────────────────────────────────────────
void test_constructor_valid() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    assert(t.getShape()[0] == 2);
    assert(t.getShape()[1] == 2);
    assert(t.getShape()[2] == 1);
    assert(t.getSize() == 4);
    std::cout << "PASSED: test_constructor_valid\n";
}

void test_constructor_invalid_shape_size() {
    try {
        Tensor t({2, 2}, {1, 2, 3, 4});
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_constructor_invalid_shape_size\n";
}

void test_constructor_invalid_shape_dimension() {
    try {
        Tensor t({2, 0, 1}, {1, 2});
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_constructor_invalid_shape_dimension\n";
}

void test_constructor_invalid_data_size() {
    try {
        Tensor t({2, 2, 1}, {1, 2, 3});
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_constructor_invalid_data_size\n";
}

// ─── at() read ────────────────────────────────────────────────────────────────
void test_at_read_first_element() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    assert(floatEq(t.at({0, 0, 0}), 1.0f));
    std::cout << "PASSED: test_at_read_first_element\n";
}

void test_at_read_last_element() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    assert(floatEq(t.at({1, 1, 0}), 4.0f));
    std::cout << "PASSED: test_at_read_last_element\n";
}

void test_at_read_middle_element() {
    Tensor t({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    assert(floatEq(t.at({1, 1, 0}), 5.0f));
    std::cout << "PASSED: test_at_read_middle_element\n";
}

void test_at_read_invalid_pos_size() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    try {
        t.at({0, 0});
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_at_read_invalid_pos_size\n";
}

void test_at_read_out_of_bounds() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    try {
        t.at({5, 0, 0});
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_at_read_out_of_bounds\n";
}

void test_at_read_negative_index() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    try {
        t.at({-1, 0, 0});
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_at_read_negative_index\n";
}

// ─── at() write ───────────────────────────────────────────────────────────────
void test_at_write() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    t.at({0, 0, 0}) = 99.0f;
    assert(floatEq(t.at({0, 0, 0}), 99.0f));
    std::cout << "PASSED: test_at_write\n";
}

void test_at_write_does_not_affect_others() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    t.at({0, 0, 0}) = 99.0f;
    assert(floatEq(t.at({1, 0, 0}), 2.0f));
    assert(floatEq(t.at({0, 1, 0}), 3.0f));
    assert(floatEq(t.at({1, 1, 0}), 4.0f));
    std::cout << "PASSED: test_at_write_does_not_affect_others\n";
}

void test_at_write_out_of_bounds() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    try {
        t.at({5, 0, 0}) = 99.0f;
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_at_write_out_of_bounds\n";
}

// ─── getShape ─────────────────────────────────────────────────────────────────
void test_getShape() {
    Tensor t2 = Tensor::zeros({3, 4, 2});
    assert(t2.getShape()[0] == 3);
    assert(t2.getShape()[1] == 4);
    assert(t2.getShape()[2] == 2);
    std::cout << "PASSED: test_getShape\n";
}

// ─── getSize ──────────────────────────────────────────────────────────────────
void test_getSize() {
    Tensor t2 = Tensor::zeros({3, 4, 2});
    assert(t2.getSize() == 24);
    std::cout << "PASSED: test_getSize\n";
}

// ─── getData ──────────────────────────────────────────────────────────────────
void test_getData_returns_correct_data() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    const std::vector<float>& data = t.getData();
    assert(floatEq(data[0], 1.0f));
    assert(floatEq(data[1], 2.0f));
    assert(floatEq(data[2], 3.0f));
    assert(floatEq(data[3], 4.0f));
    std::cout << "PASSED: test_getData_returns_correct_data\n";
}

void test_getData_size_matches_getSize() {
    Tensor t({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    assert(static_cast<int>(t.getData().size()) == t.getSize());
    std::cout << "PASSED: test_getData_size_matches_getSize\n";
}

// ─── add ──────────────────────────────────────────────────────────────────────
void test_add_basic() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({2, 1, 1}, {3, 4});
    Tensor c = a.add(b);
    assert(floatEq(c.at({0, 0, 0}), 4.0f));
    assert(floatEq(c.at({1, 0, 0}), 6.0f));
    std::cout << "PASSED: test_add_basic\n";
}

void test_add_result_shape_matches_input() {
    Tensor a({2, 3, 1}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3, 1}, {1, 2, 3, 4, 5, 6});
    Tensor c = a.add(b);
    assert(c.getShape()[0] == 2);
    assert(c.getShape()[1] == 3);
    assert(c.getShape()[2] == 1);
    std::cout << "PASSED: test_add_result_shape_matches_input\n";
}

void test_add_shape_mismatch() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({3, 1, 1}, {1, 2, 3});
    try {
        a.add(b);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_add_shape_mismatch\n";
}

// ─── dot ──────────────────────────────────────────────────────────────────────
void test_dot_basic() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({2, 1, 1}, {3, 4});
    assert(floatEq(a.dot(b), 11.0f));
    std::cout << "PASSED: test_dot_basic\n";
}

void test_dot_depth_greater_than_one() {
    Tensor a({2, 1, 2}, {1, 2, 3, 4});
    Tensor b({2, 1, 2}, {5, 6, 7, 8});
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    assert(floatEq(a.dot(b), 70.0f));
    std::cout << "PASSED: test_dot_depth_greater_than_one\n";
}

void test_dot_size_mismatch() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({3, 1, 1}, {1, 2, 3});
    try {
        a.dot(b);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_dot_size_mismatch\n";
}

// ─── hadamardProduct ──────────────────────────────────────────────────────────
void test_hadamard_basic() {
    Tensor a({2, 1, 1}, {2, 3});
    Tensor b({2, 1, 1}, {4, 5});
    Tensor c = a.hadamardProduct(b);
    assert(floatEq(c.at({0, 0, 0}), 8.0f));
    assert(floatEq(c.at({1, 0, 0}), 15.0f));
    std::cout << "PASSED: test_hadamard_basic\n";
}

void test_hadamard_shape_mismatch() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({3, 1, 1}, {1, 2, 3});
    try {
        a.hadamardProduct(b);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_hadamard_shape_mismatch\n";
}

// ─── multiply ─────────────────────────────────────────────────────────────────
void test_multiply_basic() {
    Tensor a({2, 2, 1}, {1, 2, 3, 4});
    Tensor b({2, 2, 1}, {5, 6, 7, 8});
    Tensor c = a.multiply(b);
    assert(floatEq(c.at({0, 0, 0}), 19.0f));
    assert(floatEq(c.at({1, 0, 0}), 22.0f));
    assert(floatEq(c.at({0, 1, 0}), 43.0f));
    assert(floatEq(c.at({1, 1, 0}), 50.0f));
    std::cout << "PASSED: test_multiply_basic\n";
}

void test_multiply_result_shape() {
    Tensor a({2, 2, 1}, {1, 2, 3, 4});
    Tensor b({2, 2, 1}, {5, 6, 7, 8});
    Tensor c = a.multiply(b);
    assert(c.getShape()[0] == 2);
    assert(c.getShape()[1] == 2);
    assert(c.getShape()[2] == 1);
    std::cout << "PASSED: test_multiply_result_shape\n";
}

void test_multiply_non_square_result_shape() {
    // a = 2 rows x 3 cols → {W=3, H=2}; b = 3 rows x 4 cols → {W=4, H=3}
    // valid: a.W(3) == b.H(3); result = 2 rows x 4 cols → {W=4, H=2}
    Tensor a({3, 2, 1}, {1, 2, 3, 4, 5, 6});
    Tensor b({4, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor c = a.multiply(b);
    assert(c.getShape()[0] == 4); // cols
    assert(c.getShape()[1] == 2); // rows
    assert(c.getShape()[2] == 1);
    std::cout << "PASSED: test_multiply_non_square_result_shape\n";
}

void test_multiply_depth_greater_than_one() {
    // d is {3 cols, 2 rows} per slice, e is {1 col, 3 rows} per slice
    // Valid: d's W (3) == e's H (3). Result: {1 col, 2 rows, depth 2}
    Tensor d({3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor e({1, 3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor f = d.multiply(e);
    // D=0: [[1,2,3],[4,5,6]] * [1,2,3]^T = [14, 32]
    assert(floatEq(f.at({0, 0, 0}), 14.0f));
    assert(floatEq(f.at({0, 1, 0}), 32.0f));
    // D=1: [[7,8,9],[10,11,12]] * [4,5,6]^T = [122, 167]
    assert(floatEq(f.at({0, 0, 1}), 122.0f));
    assert(floatEq(f.at({0, 1, 1}), 167.0f));
    std::cout << "PASSED: test_multiply_depth_greater_than_one\n";
}

void test_multiply_incompatible_dimensions() {
    Tensor a({3, 2, 1}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2, 1}, {1, 2, 3, 4, 5, 6});
    try {
        a.multiply(b);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_multiply_incompatible_dimensions\n";
}

void test_multiply_depth_mismatch() {
    Tensor a({2, 2, 1}, {1, 2, 3, 4});
    Tensor b({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    try {
        a.multiply(b);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_multiply_depth_mismatch\n";
}

// ─── transpose ────────────────────────────────────────────────────────────────
void test_transpose_shape() {
    Tensor t({3, 2, 1}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.transpose();
    assert(r.getShape()[0] == 2);
    assert(r.getShape()[1] == 3);
    assert(r.getShape()[2] == 1);
    std::cout << "PASSED: test_transpose_shape\n";
}

void test_transpose_preserves_size() {
    Tensor t({3, 2, 1}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.transpose();
    assert(r.getSize() == t.getSize());
    std::cout << "PASSED: test_transpose_preserves_size\n";
}

void test_transpose_values() {
    Tensor t({2, 3, 1}, {1, 2, 3, 4, 5, 6}); // 3 rows x 2 cols
    Tensor r = t.transpose();                // 2 rows x 3 cols, shape {3,2,1}
    assert(floatEq(r.at({0, 0, 0}), 1.0f));
    assert(floatEq(r.at({1, 0, 0}), 3.0f));
    assert(floatEq(r.at({2, 0, 0}), 5.0f));
    assert(floatEq(r.at({0, 1, 0}), 2.0f));
    assert(floatEq(r.at({1, 1, 0}), 4.0f));
    assert(floatEq(r.at({2, 1, 0}), 6.0f));
    std::cout << "PASSED: test_transpose_values\n";
}

// ─── flatten ──────────────────────────────────────────────────────────────────
void test_flatten_shape() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor f = t.flatten();
    assert(f.getShape()[0] == 4);
    assert(f.getShape()[1] == 1);
    assert(f.getShape()[2] == 1);
    std::cout << "PASSED: test_flatten_shape\n";
}

void test_flatten_preserves_data() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor f = t.flatten();
    assert(floatEq(f.at({0, 0, 0}), 1.0f));
    assert(floatEq(f.at({1, 0, 0}), 2.0f));
    assert(floatEq(f.at({2, 0, 0}), 3.0f));
    assert(floatEq(f.at({3, 0, 0}), 4.0f));
    std::cout << "PASSED: test_flatten_preserves_data\n";
}

void test_flatten_depth_greater_than_one() {
    Tensor t({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor f = t.flatten();
    assert(f.getShape()[0] == 8);
    assert(f.getShape()[1] == 1);
    assert(f.getShape()[2] == 1);
    assert(f.getSize() == 8);
    std::cout << "PASSED: test_flatten_depth_greater_than_one\n";
}

// ─── rotate180 ────────────────────────────────────────────────────────────────
void test_rotate180_basic() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor r = t.rotate180();
    assert(floatEq(r.at({0, 0, 0}), 4.0f));
    assert(floatEq(r.at({1, 0, 0}), 3.0f));
    assert(floatEq(r.at({0, 1, 0}), 2.0f));
    assert(floatEq(r.at({1, 1, 0}), 1.0f));
    std::cout << "PASSED: test_rotate180_basic\n";
}

void test_rotate180_preserves_shape() {
    Tensor t({3, 2, 1}, {1, 2, 3, 4, 5, 6});
    Tensor r = t.rotate180();
    assert(r.getShape()[0] == 3);
    assert(r.getShape()[1] == 2);
    assert(r.getShape()[2] == 1);
    std::cout << "PASSED: test_rotate180_preserves_shape\n";
}

void test_rotate180_twice_is_identity() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor r = t.rotate180().rotate180();
    assert(r == t);
    std::cout << "PASSED: test_rotate180_twice_is_identity\n";
}

void test_rotate180_depth_unchanged() {
    Tensor t({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor r = t.rotate180();
    assert(r.getShape()[2] == 2);
    std::cout << "PASSED: test_rotate180_depth_unchanged\n";
}

// ─── convolve ─────────────────────────────────────────────────────────────────
void test_convolve_basic_shape() {
    Tensor input({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({2, 2, 1}, {1, 0, 0, 1});
    Tensor result = input.convolve(filter, 0.0f);
    assert(result.getShape()[0] == 2);
    assert(result.getShape()[1] == 2);
    assert(result.getShape()[2] == 1);
    std::cout << "PASSED: test_convolve_basic_shape\n";
}

void test_convolve_output_depth_is_one() {
    Tensor input({3, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                             1, 2, 3, 4, 5, 6, 7, 8, 9,
                             1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({2, 2, 3}, {1, 0, 0, 1,
                              1, 0, 0, 1,
                              1, 0, 0, 1});
    Tensor result = input.convolve(filter, 0.0f);
    assert(result.getShape()[2] == 1);
    std::cout << "PASSED: test_convolve_output_depth_is_one\n";
}

void test_convolve_with_bias() {
    Tensor input({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({2, 2, 1}, {1, 0, 0, 1});
    Tensor without_bias = input.convolve(filter, 0.0f);
    Tensor with_bias = input.convolve(filter, 1.0f);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            assert(floatEq(
                with_bias.at({x, y, 0}),
                without_bias.at({x, y, 0}) + 1.0f));
        }
    }
    std::cout << "PASSED: test_convolve_with_bias\n";
}

void test_convolve_depth_mismatch() {
    Tensor input({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({2, 2, 2}, {1, 0, 0, 1, 1, 0, 0, 1});
    try {
        input.convolve(filter, 0.0f);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_convolve_depth_mismatch\n";
}

void test_convolve_invalid_stride_padding() {
    Tensor input({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({2, 2, 1}, {1, 0, 0, 1});
    try {
        input.convolve(filter, 0.0f, 4, 0);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_convolve_invalid_stride_padding\n";
}

void test_convolve_with_stride() {
    Tensor input({5, 5, 1}, {1, 2, 3, 4, 5,
                             6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15,
                             16, 17, 18, 19, 20,
                             21, 22, 23, 24, 25});
    Tensor filter({3, 3, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Tensor result = input.convolve(filter, 0.0f, 2, 0);
    assert(result.getShape()[0] == 2);
    assert(result.getShape()[1] == 2);
    std::cout << "PASSED: test_convolve_with_stride\n";
}

void test_convolve_with_padding_shape() {
    // Input 3x3, filter 3x3, padding 1, stride 1
    // {(3 - 3 + 2*1) / 1} + 1 = 3 -- same size as input
    Tensor input({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({3, 3, 1}, {0, 0, 0, 0, 1, 0, 0, 0, 0});
    Tensor result = input.convolve(filter, 0.0f, 1, 1);
    assert(result.getShape()[0] == 3);
    assert(result.getShape()[1] == 3);
    assert(result.getShape()[2] == 1);
    std::cout << "PASSED: test_convolve_with_padding_shape\n";
}

void test_convolve_values() {
    Tensor input({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({2, 2, 1}, {1, 0, 0, 1}); // picks top-left + bottom-right of each window
    Tensor result = input.convolve(filter, 0.0f);
    // (0,0): 1+5=6  (1,0): 2+6=8  (0,1): 4+8=12  (1,1): 5+9=14
    assert(floatEq(result.at({0, 0, 0}), 6.0f));
    assert(floatEq(result.at({1, 0, 0}), 8.0f));
    assert(floatEq(result.at({0, 1, 0}), 12.0f));
    assert(floatEq(result.at({1, 1, 0}), 14.0f));
    std::cout << "PASSED: test_convolve_values\n";
}

void test_convolve_padding_values() {
    // Identity filter (center=1, rest=0) with pad 1, stride 1 → output equals input
    Tensor input({3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor filter({3, 3, 1}, {0, 0, 0, 0, 1, 0, 0, 0, 0});
    Tensor result = input.convolve(filter, 0.0f, 1, 1);
    assert(result.getShape()[0] == 3 && result.getShape()[1] == 3);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
            assert(floatEq(result.at({x, y, 0}), input.at({x, y, 0})));
    std::cout << "PASSED: test_convolve_padding_values\n";
}

// ─── pad ──────────────────────────────────────────────────────────────────────
void test_pad_shape() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor p = t.pad(1);
    assert(p.getShape()[0] == 4);
    assert(p.getShape()[1] == 4);
    assert(p.getShape()[2] == 1);
    std::cout << "PASSED: test_pad_shape\n";
}

void test_pad_border_is_zero() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor p = t.pad(1);
    assert(floatEq(p.at({0, 0, 0}), 0.0f));
    assert(floatEq(p.at({3, 0, 0}), 0.0f));
    assert(floatEq(p.at({0, 3, 0}), 0.0f));
    assert(floatEq(p.at({3, 3, 0}), 0.0f));
    std::cout << "PASSED: test_pad_border_is_zero\n";
}

void test_pad_preserves_data() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor p = t.pad(1);
    assert(floatEq(p.at({1, 1, 0}), 1.0f));
    assert(floatEq(p.at({2, 1, 0}), 2.0f));
    assert(floatEq(p.at({1, 2, 0}), 3.0f));
    assert(floatEq(p.at({2, 2, 0}), 4.0f));
    std::cout << "PASSED: test_pad_preserves_data\n";
}

void test_pad_zero_returns_identical() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    Tensor p = t.pad(0);
    assert(p == t);
    std::cout << "PASSED: test_pad_zero_returns_identical\n";
}

void test_pad_invalid() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    try {
        t.pad(-1);
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_pad_invalid\n";
}

// ─── operator== ───────────────────────────────────────────────────────────────
void test_equality_same() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({2, 1, 1}, {1, 2});
    assert(a == b);
    std::cout << "PASSED: test_equality_same\n";
}

void test_equality_different_data() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({2, 1, 1}, {1, 3});
    assert(!(a == b));
    std::cout << "PASSED: test_equality_different_data\n";
}

void test_equality_different_shape() {
    Tensor a({2, 1, 1}, {1, 2});
    Tensor b({1, 2, 1}, {1, 2});
    assert(!(a == b));
    std::cout << "PASSED: test_equality_different_shape\n";
}

void test_equality_different_shape_same_size() {
    Tensor a({4, 1, 1}, {1, 2, 3, 4});
    Tensor b({2, 2, 1}, {1, 2, 3, 4});
    assert(!(a == b));
    std::cout << "PASSED: test_equality_different_shape_same_size\n";
}

// ─── operator= ────────────────────────────────────────────────────────────────
void test_assignment_valid() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    t = {5, 6, 7, 8};
    assert(floatEq(t.at({0, 0, 0}), 5.0f));
    assert(floatEq(t.at({1, 1, 0}), 8.0f));
    std::cout << "PASSED: test_assignment_valid\n";
}

void test_assignment_size_mismatch() {
    Tensor t({2, 2, 1}, {1, 2, 3, 4});
    try {
        t = {1, 2, 3};
        assert(false);
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASSED: test_assignment_size_mismatch\n";
}

// ─── zeros / ones ─────────────────────────────────────────────────────────────
void test_zeros() {
    Tensor t = Tensor::zeros({2, 2, 1});
    assert(floatEq(t.at({0, 0, 0}), 0.0f));
    assert(floatEq(t.at({1, 1, 0}), 0.0f));
    std::cout << "PASSED: test_zeros\n";
}

void test_zeros_shape() {
    Tensor t = Tensor::zeros({3, 4, 2});
    assert(t.getShape()[0] == 3);
    assert(t.getShape()[1] == 4);
    assert(t.getShape()[2] == 2);
    std::cout << "PASSED: test_zeros_shape\n";
}

void test_zeros_all_elements() {
    Tensor t = Tensor::zeros({2, 2, 2});
    const std::vector<float>& data = t.getData();
    for (int i = 0; i < t.getSize(); ++i)
        assert(floatEq(data[i], 0.0f));
    std::cout << "PASSED: test_zeros_all_elements\n";
}

void test_ones() {
    Tensor t = Tensor::ones({2, 2, 1});
    assert(floatEq(t.at({0, 0, 0}), 1.0f));
    assert(floatEq(t.at({1, 1, 0}), 1.0f));
    std::cout << "PASSED: test_ones\n";
}

void test_ones_shape() {
    Tensor t = Tensor::ones({3, 4, 2});
    assert(t.getShape()[0] == 3);
    assert(t.getShape()[1] == 4);
    assert(t.getShape()[2] == 2);
    std::cout << "PASSED: test_ones_shape\n";
}

void test_ones_all_elements() {
    Tensor t = Tensor::ones({2, 2, 2});
    const std::vector<float>& data = t.getData();
    for (int i = 0; i < t.getSize(); ++i)
        assert(floatEq(data[i], 1.0f));
    std::cout << "PASSED: test_ones_all_elements\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    // Constructor
    test_constructor_valid();
    test_constructor_invalid_shape_size();
    test_constructor_invalid_shape_dimension();
    test_constructor_invalid_data_size();

    // at() read
    test_at_read_first_element();
    test_at_read_last_element();
    test_at_read_middle_element();
    test_at_read_invalid_pos_size();
    test_at_read_out_of_bounds();
    test_at_read_negative_index();

    // at() write
    test_at_write();
    test_at_write_does_not_affect_others();
    test_at_write_out_of_bounds();

    // getShape / getSize
    test_getShape();
    test_getSize();

    // getData
    test_getData_returns_correct_data();
    test_getData_size_matches_getSize();

    // add
    test_add_basic();
    test_add_result_shape_matches_input();
    test_add_shape_mismatch();

    // dot
    test_dot_basic();
    test_dot_depth_greater_than_one();
    test_dot_size_mismatch();

    // hadamardProduct
    test_hadamard_basic();
    test_hadamard_shape_mismatch();

    // multiply
    test_multiply_basic();
    test_multiply_result_shape();
    test_multiply_non_square_result_shape();
    test_multiply_depth_greater_than_one();
    test_multiply_incompatible_dimensions();
    test_multiply_depth_mismatch();

    // transpose
    test_transpose_shape();
    test_transpose_preserves_size();
    test_transpose_values();

    // flatten
    test_flatten_shape();
    test_flatten_preserves_data();
    test_flatten_depth_greater_than_one();

    // rotate180
    test_rotate180_basic();
    test_rotate180_preserves_shape();
    test_rotate180_twice_is_identity();
    test_rotate180_depth_unchanged();

    // convolve
    test_convolve_basic_shape();
    test_convolve_values();
    test_convolve_output_depth_is_one();
    test_convolve_with_bias();
    test_convolve_depth_mismatch();
    test_convolve_invalid_stride_padding();
    test_convolve_with_stride();
    test_convolve_with_padding_shape();
    test_convolve_padding_values();

    // pad
    test_pad_shape();
    test_pad_border_is_zero();
    test_pad_preserves_data();
    test_pad_zero_returns_identical();
    test_pad_invalid();

    // operator==
    test_equality_same();
    test_equality_different_data();
    test_equality_different_shape();
    test_equality_different_shape_same_size();

    // operator=
    test_assignment_valid();
    test_assignment_size_mismatch();

    // zeros / ones
    test_zeros();
    test_zeros_shape();
    test_zeros_all_elements();
    test_ones();
    test_ones_shape();
    test_ones_all_elements();

    std::cout << "\nAll tests passed\n";
    return 0;
}