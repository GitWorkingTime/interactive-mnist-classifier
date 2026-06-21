#ifndef MNIST_H
#define MNIST_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Function Declarations ───────────────────────────────────────────────────
namespace mnist {

/**
 * @brief Reverses the byte order of a 32-bit integer (big-endian ↔ little-endian).
 *
 * @par Description
 * MNIST IDX files store their header integers in big-endian order. On a
 * little-endian host these must be byte-swapped to be read correctly. This
 * swaps the four bytes of a 32-bit integer end-for-end.
 *
 * @param num The integer to byte-swap (as read directly from the file).
 * @return The same integer with its byte order reversed.
 */
int reverseInt(int num);

/**
 * @brief Loads MNIST image data from an IDX image file into a list of Tensors.
 *
 * @par Description
 * Reads an IDX3 image file, validates its magic number (2051), and converts each
 * 28x28 image into a Tensor of shape {cols, rows, 1}. Pixel values (0–255) are
 * normalized to the range [0, 1] by dividing by 255.
 *
 * @param path Filesystem path to the IDX image file (e.g. "train-images-idx3-ubyte").
 *
 * @return A vector of Tensors, one per image, each with shape {cols, rows, 1}.
 *
 * @throws std::runtime_error if the file cannot be opened.
 * @throws std::runtime_error if the magic number is not 2051 (not a valid IDX image file).
 */
std::vector<Tensor> loadImages(const std::string& path, bool verbose = false);

/**
 * @brief Loads MNIST label data from an IDX label file.
 *
 * @par Description
 * Reads an IDX1 label file, validates its magic number (2049), and returns the
 * label for each image as an integer in the range [0, 9].
 *
 * @param path Filesystem path to the IDX label file (e.g. "train-labels-idx1-ubyte").
 *
 * @return A vector of integer labels, one per image, each in [0, 9].
 *
 * @throws std::runtime_error if the file cannot be opened.
 * @throws std::runtime_error if the magic number is not 2049 (not a valid IDX label file).
 */
std::vector<int> loadLabels(const std::string& path, bool verbose = false);

/**
 * @brief Converts a single integer label into a one-hot encoded target Tensor.
 *
 * @par Description
 * Maps a digit label (0–9) to a {10, 1, 1} Tensor that is 1.0 at the index of
 * the label and 0.0 elsewhere. Used to build the target distribution for
 * cross-entropy loss during training.
 *
 * @param label The digit label, expected in [0, 9].
 *
 * @return A Tensor of shape {10, 1, 1}, one-hot encoded.
 *
 * @throws std::invalid_argument if label is not within [0, 9].
 *
 * @par Example
 * @code
 * Tensor v = mnist::oneHotEncodeLabels(3);
 * // v data = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, shape {10, 1, 1}
 * @endcode
 */
Tensor oneHotEncodeLabels(int label);

} // namespace mnist

#endif