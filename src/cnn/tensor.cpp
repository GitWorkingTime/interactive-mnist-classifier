// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"

// ─── Class Definition ────────────────────────────────────────────────────────
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

const float Tensor::at(const std::vector<int>& pos) const {
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