// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"

// ─── Helpers ─────────────────────────────────────────────────────────────────
void compareDimensions(const std::vector<int>& curr, const std::vector<int>& other) {
    if (curr[0] != other[0] || curr[1] != other[1] || curr[2] != other[2]) {
        std::string shapeStr = "{" + std::to_string(curr[0]) + ", " + std::to_string(curr[1]) + ", " + std::to_string(curr[2]) + "}";
        std::string otherShapeStr = "{" + std::to_string(other[0]) + ", " + std::to_string(other[1]) + ", " + std::to_string(other[2]) + "}";
        throw std::invalid_argument("ERROR: Tensor dimensions do not match - This: " + shapeStr + " |Other: " + otherShapeStr);
    }
    return;
}

// ─── Class Definition ────────────────────────────────────────────────────────
// Constructor
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) {
    // Verify valid shape dimensions
    if (shape.size() != 3) {
        throw std::invalid_argument("ERROR: Shape dimensions does not follow {W, H, D}.");
    } else if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
        std::string errorMsg = "ERROR: Shape dimension(s) is below or equal to 0 - {" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + "}";
        throw std::invalid_argument(errorMsg);
    }

    // Verify data fits shape dimensions
    const int len = shape[0] * shape[1] * shape[2];
    if (len != static_cast<int>(data.size())) {
        throw std::invalid_argument("ERROR: Tensor data does not match Tensor shape");
    }

    // Initialize private member variables
    this->shape = shape;
    this->data = data;
}

// ─── Display ─────────────────────────────────────────────────────────────────
void Tensor::displayASCII() const {
    const std::string chars = " .:-=+*#%@";
    const int charCount = static_cast<int>(chars.size() - 1);

    auto minmax = std::minmax_element(data.begin(), data.end());
    float minVal = *minmax.first;
    float maxVal = *minmax.second;
    float range = maxVal - minVal;

    // Pre-allocate buffer
    // Each pixel = 2 chars, each row = W*2 + 1 newline, each slice = H rows + 1 newline + header
    std::string buffer;
    buffer.reserve(shape[2] * (shape[1] * (shape[0] * 2 + 1) + 10));

    for (int z = 0; z < shape[2]; ++z) {
        buffer += "D = ";
        buffer += std::to_string(z);
        buffer += "\n";

        for (int y = 0; y < shape[1]; ++y) {
            for (int x = 0; x < shape[0]; ++x) {
                // Formula the same as the 'at()' function
                int index = x + (y * shape[0]) + (z * shape[0] * shape[1]);

                // Normalization
                float value = 0.0f;
                if (range > 0) {
                    // Formula: (x - min) / (max - min)
                    value = (data[index] - minVal) / range;
                }
                char c = chars[static_cast<int>(value * charCount)];
                buffer += c;
                buffer += c;
            }
            buffer += "\n";
        }
        buffer += "\n";
    }
    std::cout << buffer;
}

void Tensor::displayRaw() const {
    // Find the widest number for column alignment
    std::string buffer;
    int colWidth = 0;

    for (float val : data) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << val;
        int width = static_cast<int>(oss.str().size());
        if (width > colWidth)
            colWidth = width;
    }

    // Total row width = (colWidth + 1) * W + 3 for borders
    int rowWidth = (colWidth + 1) * shape[0] + 3;

    // Pre-allocate buffer
    buffer.reserve(shape[2] * (shape[1] * (rowWidth + 1) + 3));

    for (int z = 0; z < shape[2]; ++z) {
        buffer += "D = ";
        buffer += std::to_string(z);
        buffer += "\n";

        // Top border
        buffer += "\u250C";
        for (int i = 0; i < rowWidth - 2; ++i)
            buffer += " ";
        buffer += "\u2510";
        buffer += "\n";

        // Rows
        for (int y = 0; y < shape[1]; ++y) {
            buffer += "\u2502";
            for (int x = 0; x < shape[0]; ++x) {
                int index = x + (y * shape[0]) + (z * shape[0] * shape[1]);

                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2) << std::setw(colWidth) << data[index];
                buffer += " ";
                buffer += oss.str();
            }
            buffer += " \u2502";
            buffer += "\n";
        }

        // Bottom border
        buffer += "\u2514";
        for (int i = 0; i < rowWidth - 2; ++i)
            buffer += " ";
        buffer += "\u2518";
        buffer += "\n\n";
    }

    std::cout << buffer;
}

// ─── Read/Write ───────────────────────────────────────────────────────────
float Tensor::at(const std::vector<int>& pos) const {
    // Verify the pos arg size matches
    if (pos.size() != 3) {
        throw std::invalid_argument("ERROR: pos arg does not follow {W, H, D} format");

        // Verify the pos arg dimensions are in bound
    } else if (pos[0] < 0 || pos[0] >= shape[0] ||
               pos[1] < 0 || pos[1] >= shape[1] ||
               pos[2] < 0 || pos[2] >= shape[2]) {
        std::string posStr = "{" + std::to_string(pos[0]) + ", " + std::to_string(pos[1]) + ", " + std::to_string(pos[2]) + "}";
        std::string shapeStr = "{" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + "}";
        throw std::invalid_argument("ERROR: pos out of bounds - pos: " + posStr + " | shape: " + shapeStr);
    }

    int index = pos[0] + (pos[1] * shape[0]) + (pos[2] * shape[0] * shape[1]);
    return data[index];
}

// Returns reference to the data value which allows for modification
float& Tensor::at(const std::vector<int>& pos) {
    // Verify the pos arg size matches
    if (pos.size() != 3) {
        throw std::invalid_argument("ERROR: pos arg does not follow {W, H, D} format");

        // Verify the pos arg dimensions are in bound
    } else if (pos[0] < 0 || pos[0] >= shape[0] ||
               pos[1] < 0 || pos[1] >= shape[1] ||
               pos[2] < 0 || pos[2] >= shape[2]) {
        std::string posStr = "{" + std::to_string(pos[0]) + ", " + std::to_string(pos[1]) + ", " + std::to_string(pos[2]) + "}";
        std::string shapeStr = "{" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + "}";
        throw std::invalid_argument("ERROR: pos out of bounds - pos: " + posStr + " | shape: " + shapeStr);
    }

    int index = pos[0] + (pos[1] * shape[0]) + (pos[2] * shape[0] * shape[1]);
    return data[index];
}

const std::vector<int>& Tensor::getShape() const {
    return shape;
}

const std::vector<float>& Tensor::getData() const {
    return data;
}

int Tensor::getSize() const {
    return data.size();
}

// ─── Lin. Alg Operations ──────────────────────────────────────────────────
Tensor Tensor::add(const Tensor& tensor) const {
    // Verify the dimensions are equal
    compareDimensions(shape, tensor.getShape());

    const std::vector<float>& otherData = tensor.getData();
    std::vector<float> sum = {};
    sum.reserve(data.size());

    // Best case we can do is O(n) time-complexity
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        sum.push_back(data[i] + otherData[i]);
    }

    return Tensor(shape, sum);
}

float Tensor::dot(const Tensor& tensor) const {
    const std::vector<float>& otherData = tensor.getData();

    // Verify the data length are equal
    if (otherData.size() != data.size()) {
        throw std::invalid_argument("ERROR: Tensor data length do not match");
    }

    float sum = 0.0f;

    // Best case we can do is O(n) time-complexity
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        sum += otherData[i] * data[i];
    }

    return sum;
}

Tensor Tensor::hadamardProduct(const Tensor& tensor) const {
    // Verify the dimensions are equal
    compareDimensions(shape, tensor.getShape());

    // Compute the element-wise multiplication
    const std::vector<float>& otherData = tensor.getData();
    std::vector<float> result = {};
    result.reserve(data.size());

    // Best case we can do is O(n) time-complexity
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        result.push_back(data[i] * otherData[i]);
    }

    return Tensor(shape, result);
}

Tensor Tensor::multiply(const Tensor& tensor) const {
    // Verify the dimensions allows for matrix multiplication
    std::vector<int> otherShape = tensor.getShape();
    if (shape[0] != otherShape[1] || shape[2] != otherShape[2]) {
        std::string shapeStr = "{" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + "}";
        std::string otherShapeStr = "{" + std::to_string(otherShape[0]) + ", " + std::to_string(otherShape[1]) + ", " + std::to_string(otherShape[2]) + "}";
        throw std::invalid_argument("ERROR: Tensor dimensions do not allow for matrix multiplication - This: " + shapeStr + " |Other: " + otherShapeStr);
    }

    // Dimensions (W = columns, H = rows)
    const int thisCols = shape[0];
    const int thisRows = shape[1];
    const int depth = shape[2];
    const int otherCols = otherShape[0];
    const int otherRows = otherShape[1];

    // Result has this tensor's rows and the other tensor's columns
    std::vector<float> resultData(otherCols * thisRows * depth, 0.0f);

    // The following loops uses pointer arithmetic!

    // Returns the address of data's first element
    const float* thisValues = data.data();

    // Returns the address of the other tensor's first element
    const float* otherValues = tensor.getData().data();

    // Returns the address of the result's first element
    float* resultValues = resultData.data();

    for (int z = 0; z < depth; ++z) {
        // Start of depth slice z in each tensor
        const float* thisSlice = thisValues + z * thisCols * thisRows;
        const float* otherSlice = otherValues + z * otherCols * otherRows;
        float* resultSlice = resultValues + z * otherCols * thisRows;

        for (int row = 0; row < thisRows; ++row) {
            const float* thisRow = thisSlice + row * thisCols;
            float* resultRow = resultSlice + row * otherCols;

            // Result row = weighted sum of the other tensor's rows,
            // weights taken from this tensor's row
            for (int k = 0; k < thisCols; ++k) {
                const float weight = thisRow[k];
                const float* otherRow = otherSlice + k * otherCols;

                for (int col = 0; col < otherCols; ++col)
                    resultRow[col] += weight * otherRow[col];
            }
        }
    }

    return Tensor({otherCols, thisRows, depth}, resultData);
}

Tensor Tensor::transpose() const {
    const int W = shape[0], H = shape[1], D = shape[2];
    std::vector<float> result(data.size());

    // Re-order the tensor
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                // element (x,y) → (y,x); new width is H
                result[y + x * H + z * W * H] = data[x + y * W + z * W * H];
            }
        }
    }

    return Tensor({H, W, D}, result);
}

// Each output cell is a dot product of the filter against the input window
// at that position, summed across all channels (depth collapses to 1).
// Pointers are hoisted per loop level (slice → row) to avoid recomputing
// offsets in the hot inner loop. See multiply() for the same pattern.
Tensor Tensor::convolve(const Tensor& filter, float bias, std::size_t stride, std::size_t padding) const {
    // Validate dimensions
    std::vector<int> filterShape = filter.getShape();

    if (filterShape[2] != shape[2]) {
        std::string errorMsg = "ERROR: filter's and this tensor's depths do not match - filter: " + std::to_string(filterShape[2]) + " this: " + std::to_string(shape[2]);
        throw std::invalid_argument(errorMsg);
    }

    // Validate Padding and stride
    int numerW = shape[0] - filterShape[0] + 2 * static_cast<int>(padding);
    int numerH = shape[1] - filterShape[1] + 2 * static_cast<int>(padding);
    int s = static_cast<int>(stride);

    if (numerW < 0 || numerH < 0 || numerW % s != 0 || numerH % s != 0) {
        throw std::invalid_argument("ERROR: stride and/or padding does not result in a positive integer for the final tensor size");
    }
    int finalWidth = numerW / s + 1;
    int finalHeight = numerH / s + 1;

    // Initialize with a padded Tensor
    Tensor padded = this->pad(static_cast<int>(padding));

    // Compute the convolution:
    const std::vector<float>& filterData = filter.getData();
    std::vector<int> paddedShape = padded.getShape();
    const std::vector<float>& paddedData = padded.getData();
    std::vector<float> result(finalWidth * finalHeight, 0.0f);

    // Similar pointer arithmetic as with multiply()
    const float* pData = paddedData.data();
    const float* fData = filterData.data();
    const int padW = paddedShape[0], padH = paddedShape[1];
    const int fW = filterShape[0], fH = filterShape[1];

    for (int outputY = 0; outputY < finalHeight; ++outputY) {
        for (int outputX = 0; outputX < finalWidth; ++outputX) {
            float sum = bias;
            for (int z = 0; z < shape[2]; ++z) {
                const float* pSlice = pData + z * padW * padH; // channel z of input
                const float* fSlice = fData + z * fW * fH;     // channel z of filter
                for (int filterY = 0; filterY < fH; ++filterY) {
                    const int innerY = outputY * s + filterY;
                    const float* pRow = pSlice + innerY * padW; // the input row
                    const float* fRow = fSlice + filterY * fW;  // the filter row
                    const int baseX = outputX * s;
                    for (int filterX = 0; filterX < fW; ++filterX)
                        sum += pRow[baseX + filterX] * fRow[filterX]; // contiguous-ish
                }
            }
            result[outputX + outputY * finalWidth] = sum;
        }
    }

    return Tensor({finalWidth, finalHeight, 1}, result);
}

Tensor Tensor::rotate180() const {
    std::vector<float> result = data; // copy, same shape
    const int sliceSize = shape[0] * shape[1];
    for (int z = 0; z < shape[2]; ++z) {
        auto begin = result.begin() + z * sliceSize;
        std::reverse(begin, begin + sliceSize); // reverse just this slice
    }
    return Tensor(shape, result);
}

Tensor Tensor::flatten() const {
    return Tensor({getSize(), 1, 1}, data);
}

Tensor Tensor::pad(int padding) const {
    // Verify padding
    if (padding < 0) {
        throw std::invalid_argument("ERROR: Padding is < 0");
    }

    // Return early if padding is 0
    if (padding == 0) {
        return *this;
    }

    // Initialize a buffer Tensor with just zeros
    Tensor buffer = zeros({shape[0] + 2 * padding, shape[1] + 2 * padding, shape[2]});

    // The padding acts as a shift, pushing the original value to the middle
    for (int z = 0; z < shape[2]; ++z) {
        for (int y = 0; y < shape[1]; ++y) {
            for (int x = 0; x < shape[0]; ++x) {
                buffer.at({x + padding, y + padding, z}) = this->at({x, y, z});
            }
        }
    }

    return buffer;
}

// // ─── Operator Overloading ─────────────────────────────────────────────────
Tensor& Tensor::operator=(const std::vector<float>& data) {
    // Validate new data's # of elements
    int expected = shape[0] * shape[1] * shape[2];
    if (static_cast<int>(data.size()) != expected) {
        throw std::invalid_argument(
            "ERROR: data size does not match tensor shape - expected: " +
            std::to_string(expected) + " got: " + std::to_string(data.size()));
    }

    // Replace data
    this->data = data;
    return *this;
}

bool Tensor::operator==(const Tensor& tensor) const {
    // Compare both shape and data
    return shape == tensor.shape && data == tensor.data;
}

// // ─── Pre-defined Tensors ──────────────────────────────────────────────────
Tensor Tensor::zeros(const std::vector<int>& shape) {
    // Validate shape's size:
    if (shape.size() != 3) {
        throw std::invalid_argument("ERROR: the given shape for zeros does not match the format {W, H, D}");
    }

    std::vector<float> tensorData(shape[0] * shape[1] * shape[2], 0.0f);
    return Tensor(shape, tensorData);
}

Tensor Tensor::ones(const std::vector<int>& shape) {
    // Validate shape's size:
    if (shape.size() != 3) {
        throw std::invalid_argument("ERROR: the given shape for ones does not match the format {W, H, D}");
    }

    std::vector<float> tensorData(shape[0] * shape[1] * shape[2], 1.0f);
    return Tensor(shape, tensorData);
}