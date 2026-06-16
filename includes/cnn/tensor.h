#ifndef TENSOR_H
#define TENSOR_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Class Declaration ───────────────────────────────────────────────────────
/**
 * @brief A 3D Tensor class backed by a 1D float array
 *
 * @par Description
 * Represents a 3D Tensor with the shape {W, H, D}. All values
 * are stored internally as a 1D array in row-major order
 *
 * @par Conventions
 * - Shape format: {W (width), H (height), D (depth)}
 * - Shape bounds (1-based): [1, +infinity)
 * - Position (0-based indexing): [0, dimension)
 *
 * @par Example
 * @code
 * Tensor t({2, 2, 1}, {1, 2, 3, 4});
 * float val = t.at({0, 0, 0}); // returns 1
 * @endcode
 */
class Tensor {
private:
    /**
     * @brief flat 1D array storing all tensor values in row-major order
     */
    std::vector<float> data;

    /**
     * @brief flat 1D array storing the tensor's shape dimensions in {W, H, D} order
     */
    std::vector<int> shape;

public:
    // ─── Constructor ──────────────────────────────────────────────────────────
    /**
     * @brief Constructs a Tensor with a given shape and data.
     *
     * @par Description
     * Allocates a 3D tensor with the specified shape and initializes
     * it with the provided float values in row-major order.
     *
     * @param shape Dimensions in {W, H, D} format, each >= 1
     * @param values Float data, must have exactly W*H*D elements
     *
     * @throws std::invalid_argument if shape does not have exactly 3 dimensions
     * @throws std::invalid_argument if any shape dimension is < 1
     * @throws std::invalid_argument if values.size() != W*H*D
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * @endcode
     */
    Tensor(const std::vector<int>& shape, const std::vector<float>& values);

    // ─── Display (in terminal) ────────────────────────────────────────────────
    /**
     * @brief Displays the Tensor as ASCII art in the terminal.
     *
     * @par Description
     * Maps each float value in [0, 1] to an ASCII character to produce
     * a visual representation of the tensor. If D > 1, each depth slice
     * is displayed separately.
     *
     * @note Values are normalized to [0, 1] if they are not already
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {0.0, 0.5, 0.8, 1.0});
     * t.displayASCII();
     * //   D = 0:
     * //    =
     * //   *@
     * @endcode
     */
    void displayASCII() const;

    /**
     * @brief Displays the Tensor as one or more 2D grids in the terminal.
     *
     * @par Description
     * Prints each depth slice as a 2D grid of raw float values.
     * If D > 1, slices are displayed sequentially labeled by depth index.
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
     * t.displayRaw();
     * //   D = 0:
     * //   ┌     ┐
     * //   │ 1 2 │
     * //   │ 3 4 │
     * //   └     ┘
     * //
     * //   D = 1:
     * //   ┌     ┐
     * //   │ 5 6 │
     * //   │ 7 8 │
     * //   └     ┘
     * @endcode
     */
    void displayRaw() const;

    // ─── Read/Write ───────────────────────────────────────────────────────────
    /**
     * @brief Returns the value at a given position.
     *
     * @par Description
     * Retrieves the float value stored at the given 3D position
     * using 0-based indexing in {W, H, D} format.
     *
     * @param pos Position in {W, H, D} format, 0-based.
     *            Each dimension must satisfy 0 <= pos[i] < shape[i].
     *
     * @return The float value at the given position.
     *
     * @throws std::invalid_argument if pos does not have exactly 3 dimensions
     * @throws std::invalid_argument if pos is out of bounds
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * float val = t.at({1, 0, 0}); // returns 2
     * @endcode
     */
    float at(const std::vector<int>& pos) const;

    /**
     * @brief Writes the value at a given position.
     *
     * @par Description
     * Writes a float value stored at the given 3D position
     * using 0-based indexing in {W, H, D} format.
     *
     * @param pos Position in {W, H, D} format, 0-based.
     *            Each dimension must satisfy 0 <= pos[i] < shape[i].
     *
     * @throws std::invalid_argument if pos does not have exactly 3 dimensions
     * @throws std::invalid_argument if pos is out of bounds
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * t.at({0, 0, 0}) = 5; // Data is now {5, 2, 3, 4}
     * @endcode
     */
    float& at(const std::vector<int>& pos);

    /**
     * @brief Returns the shape of the Tensor.
     *
     * @par Description
     * Returns the shape dimensions in {W, H, D} format.
     * Each dimension is guaranteed to be >= 1.
     *
     * @return A const reference to the shape vector in {W, H, D} format.
     *
     * @par Example
     * @code
     * Tensor t({2, 3, 1}, {1, 2, 3, 4, 5, 6});
     * t.getShape(); // returns {2, 3, 1}
     * @endcode
     */
    const std::vector<int>& getShape() const;

    /**
     * @brief Returns the raw float data of the Tensor.
     *
     * @par Description
     * Returns the underlying 1D float array in row-major order.
     * Element order follows: x + y*W + z*W*H.
     *
     * @return A const reference to the internal data vector.
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * t.getData(); // returns {1, 2, 3, 4}
     * @endcode
     */
    const std::vector<float>& getData() const;

    /**
     * @brief Returns the total number of elements in the Tensor.
     *
     * @par Description
     * Equivalent to W * H * D.
     *
     * @return Total element count as an int.
     *
     * @par Example
     * @code
     * Tensor t({2, 3, 1}, {1, 2, 3, 4, 5, 6});
     * t.getSize(); // returns 6
     * @endcode
     */
    int getSize() const;

    // ─── Lin. Alg Operations ──────────────────────────────────────────────────
    /**
     * @brief Adds two Tensors element-wise.
     *
     * @par Description
     * Performs element-wise addition of this Tensor and the given Tensor.
     * Both tensors must have identical shapes.
     *
     * @param tensor The Tensor to add.
     *
     * @return A new Tensor containing the element-wise sum.
     *
     * @throws std::invalid_argument if shapes do not match
     *
     * @par Example
     * @code
     * Tensor a({2, 1, 1}, {1, 2});
     * Tensor b({2, 1, 1}, {3, 4});
     * Tensor c = a.add(b); // {4, 6}
     * @endcode
     */
    Tensor add(const Tensor& tensor) const;

    /**
     * @brief Computes the dot product of two Tensors.
     *
     * @par Description
     * Flattens both tensors and computes the sum of element-wise products.
     * Both tensors must have the same total number of elements.
     *
     * @param tensor The Tensor to dot with.
     *
     * @return A float scalar representing the dot product.
     *
     * @throws std::invalid_argument if total element counts do not match
     *
     * @par Example
     * @code
     * Tensor a({2, 1, 1}, {1, 2});
     * Tensor b({2, 1, 1}, {3, 4});
     * float result = a.dot(b); // returns 11
     * @endcode
     */
    float dot(const Tensor& tensor) const;

    /**
     * @brief Computes the Hadamard (element-wise) product of two Tensors.
     *
     * @par Description
     * Multiplies corresponding elements of two Tensors together.
     * Both tensors must have identical shapes.
     *
     * @param tensor The Tensor to multiply element-wise with.
     *
     * @return A new Tensor containing the element-wise products.
     *
     * @throws std::invalid_argument if shapes do not match
     *
     * @par Example
     * @code
     * Tensor a({2, 1, 1}, {2, 3});
     * Tensor b({2, 1, 1}, {4, 5});
     * Tensor c = a.hadamardProduct(b); // {8, 15}
     * @endcode
     */
    Tensor hadamardProduct(const Tensor& tensor) const;

    /**
     * @brief Multiplies two Tensors using matrix multiplication.
     *
     * @par Description
     * Performs matrix multiplication on the W and H dimensions of the Tensor.
     * The W of this Tensor must equal the H of the given Tensor.
     * D must be equal across both Tensors.
     *
     * @param tensor The Tensor to multiply with.
     *
     * @return A new Tensor containing the matrix product.
     *
     * @throws std::invalid_argument if dimensions are incompatible
     *
     * @note W represents columns and H represents rows.
     *       For multiplication to be valid, this Tensor's W must be equal
     *       to the given Tensor's H. Both Tensor's depths must be equal.
     *
     * @par Example
     * @code
     * Tensor a({2, 2, 1}, {1, 2, 3, 4});
     * Tensor b({2, 2, 1}, {5, 6, 7, 8});
     * Tensor c = a.multiply(b);
     * //   {19, 22
     * //    43, 50}
     *
     * Tensor d({3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
     * Tensor e({1, 3, 2}, {1, 2, 3, 4, 5, 6});
     * Tensor f = d.multiply(e);
     * //   D = 0:
     * //   {14,
     * //    32}
     * //
     * //   D = 1:
     * //   {122,
     * //    167}
     * @endcode
     */
    Tensor multiply(const Tensor& tensor) const;

    /**
     * @brief Transposes the Tensor along the W and H dimensions.
     *
     * @par Description
     * Swaps the W and H dimensions of the Tensor. The resulting Tensor
     * has shape {H, W, D}. D is unchanged.
     *
     * @return A new Tensor with W and H dimensions swapped.
     *
     * @par Example
     * @code
     * Tensor t({3, 2, 1}, {1, 2, 3, 4, 5, 6});
     * Tensor r = t.transpose(); // shape {2, 3, 1}
     * @endcode
     */
    Tensor transpose() const;

    /**
     * @brief Convolves the Tensor with a given filter.
     *
     * @par Synopsis
     * Tensor convolve(const Tensor& filter, float bias, std::size_t stride = 1, std::size_t padding = 0) const
     *
     * @par Description
     * Performs a 2D convolution across the W and H dimensions using the
     * given filter, bias, stride, and padding. The filter's D must match
     * this Tensor's D. Produces an output Tensor with D = 1.
     *
     * @param filter  The convolution filter, must have same D as this Tensor
     * @param bias    Scalar bias added to each output element
     * @param stride  Step size of the filter (default: 1)
     * @param padding Zero-padding applied to W and H borders (default: 0)
     *
     * @return A new Tensor containing the convolution result with D = 1.
     *
     * @throws std::invalid_argument if filter D does not match this Tensor's D
     * @throws std::invalid_argument if stride and/or padding does not result in a positive integer for the final tensor size
     *
     * @note The final width and height are calculated using this formula: {(W - F + 2P) / S} + 1
     * where:
     * - W: input volume size
     * - F: receptive field size
     * - S: stride
     * - P: padding
     *
     * For example: 7x7 input with a 3x3 filter, stride = 1, padding = 0.
     * W (width)  - [(7 - 3 + 2(0)) / 1] + 1 => [(4) / 1] + 1 => 5
     * H (height) - [(7 - 3 + 2(0)) / 1] + 1 => [(4) / 1] + 1 => 5
     * We would get a 5x5 result
     *
     * If the strides and/or padding results is not a positive integer, the parameters are invalid
     *
     * @par Example
     * @code
     * Tensor input({3, 3, 1}, {1,2,3,4,5,6,7,8,9});
     * Tensor filter({2, 2, 1}, {1,0,0,1});
     * Tensor result = input.convolve(filter, 0.0f); // shape {2, 2, 1}
     * @endcode
     */
    Tensor convolve(const Tensor& filter, float bias, std::size_t stride = 1, std::size_t padding = 0) const;

    /**
     * @brief Rotates the Tensor 180 degrees along the W and H dimensions.
     *
     * @par Description
     * Reverses the order of elements along both the W and H dimensions.
     * Commonly used in backpropagation to compute the gradient of a
     * convolution. D is unchanged.
     *
     * @return A new Tensor rotated 180 degrees.
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * Tensor r = t.rotate180(); // {4, 3, 2, 1}
     * @endcode
     */
    Tensor rotate180() const;

    /**
     * @brief Flattens the Tensor to shape {W*H*D, 1, 1}.
     *
     * @par Description
     * Reshapes the Tensor into a 1D column vector. Commonly used
     * when transitioning from convolutional layers to fully connected layers.
     *
     * @return A new Tensor with shape {W*H*D, 1, 1}.
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * Tensor f = t.flatten(); // shape {4, 1, 1}
     * @endcode
     */
    Tensor flatten() const;

    /**
     * @brief Returns a zero-padded copy of the Tensor.
     *
     * @par Description
     * Adds P layers of zeros around the W and H dimensions.
     * Commonly used before convolution to preserve spatial dimensions.
     * D is unchanged.
     *
     * @param padding Number of zero layers to add around W and H borders.
     *
     * @return A new Tensor with shape {W+2P, H+2P, D}.
     *
     * @throws std::invalid_argument if padding < 0
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * Tensor p = t.pad(1); // shape {4, 4, 1}
     * //   0 0 0 0
     * //   0 1 2 0
     * //   0 3 4 0
     * //   0 0 0 0
     * @endcode
     */
    Tensor pad(int padding) const;

    // ─── Operator Overloading ─────────────────────────────────────────────────
    /**
     * @brief Assigns a flat float array to the Tensor's data.
     *
     * @par Description
     * Replaces the internal data of the Tensor with the given float vector.
     * The size of the new data must match the existing W*H*D element count.
     *
     * @param data A flat float vector in row-major order.
     *
     * @return A reference to this Tensor.
     *
     * @throws std::invalid_argument if data.size() != W*H*D
     *
     * @par Example
     * @code
     * Tensor t({2, 2, 1}, {1, 2, 3, 4});
     * t = {5, 6, 7, 8};
     * @endcode
     */
    Tensor& operator=(const std::vector<float>& data);

    /**
     * @brief Returns true if two Tensors have identical shape and data.
     *
     * @param tensor The Tensor to compare against.
     *
     * @return True if shape and all data elements are equal, false otherwise.
     *
     * @par Example
     * @code
     * Tensor a({2, 1, 1}, {1, 2});
     * Tensor b({2, 1, 1}, {1, 2});
     * assert(a == b); // true
     * @endcode
     */
    bool operator==(const Tensor& tensor) const;

    // ─── Pre-defined Tensors ──────────────────────────────────────────────────
    /**
     * @brief Returns a Tensor of the given shape filled with zeros.
     *
     * @param shape Dimensions in {W, H, D} format, each >= 1
     *
     * @return A new Tensor with all elements set to 0.0f
     *
     * @throws std::invalid_argument if the given shape does not match the {W, H, D} format
     *
     * @par Example
     * @code
     * Tensor t = Tensor::zeros({2, 2, 1}); // {0, 0, 0, 0}
     * @endcode
     */
    static Tensor zeros(const std::vector<int>& shape);

    /**
     * @brief Returns a Tensor of the given shape filled with ones.
     *
     * @param shape Dimensions in {W, H, D} format, each >= 1
     *
     * @return A new Tensor with all elements set to 1.0f
     *
     * @throws std::invalid_argument if the given shape does not match the {W, H, D} format
     *
     * @par Example
     * @code
     * Tensor t = Tensor::ones({2, 2, 1}); // {1, 1, 1, 1}
     * @endcode
     */
    static Tensor ones(const std::vector<int>& shape);
};

#endif