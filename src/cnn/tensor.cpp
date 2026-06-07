// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"

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
    if (len != data.size()) {
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