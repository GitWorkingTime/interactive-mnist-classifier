// ─── Imports ─────────────────────────────────────────────────────────────────
#include "pooling.h"

// ─── Function Definitions ────────────────────────────────────────────────────
pooling::PoolResult pooling::maxPooling(const Tensor& tensor, int poolSize, int stride) {
    // Verify stride is greater than 0
    if (stride <= 0) {
        throw std::invalid_argument("ERROR: stride is less than 0. Ensure it is a positive integer!\nStride: " + std::to_string(stride));
    }

    // Verify the poolSize and stride works with the tensor's shape
    const std::vector<int>& tensorShape = tensor.getShape();

    int numerW = tensorShape[0] - poolSize;
    int numerH = tensorShape[1] - poolSize;

    if ((numerW % stride != 0) || (numerH % stride != 0) || (numerW < 0) || (numerH < 0)) {
        throw std::invalid_argument("ERROR: parameters result in a non-integer shape\nWidth: " + std::to_string(numerW) + "/" + std::to_string(stride) + "\nHeight: " + std::to_string(numerH) + "/" + std::to_string(stride));
    }

    // max pooled Tensor's shape
    int finalW = (numerW / stride) + 1;
    int finalH = (numerH / stride) + 1;

    // Compute max pooling and argmax
    const std::vector<float>& tensorData = tensor.getData();
    std::vector<int> argmax = {};
    argmax.reserve(finalW * finalH * tensorShape[2]);

    std::vector<float> data = {};
    data.reserve(finalW * finalH * tensorShape[2]);

    for (int z = 0; z < tensorShape[2]; ++z) {
        // Window steps
        for (int oy = 0; oy < finalH; ++oy) {
            for (int ox = 0; ox < finalW; ++ox) {

                // Negative infinity so that any value afterwards is deemed "max"
                float maxVal = -std::numeric_limits<float>::infinity();
                int maxIdx = -1;

                // Inside window
                for (int iy = 0; iy < poolSize; ++iy) {
                    for (int ix = 0; ix < poolSize; ++ix) {
                        int inX = ox * stride + ix;
                        int inY = oy * stride + iy;
                        int flatIdx = inX + inY * tensorShape[0] + z * tensorShape[0] * tensorShape[1];

                        if (tensorData[flatIdx] > maxVal) { // strict > → first max wins ties
                            maxVal = tensorData[flatIdx];
                            maxIdx = flatIdx;
                        }
                    }
                }

                data.push_back(maxVal);
                argmax.push_back(maxIdx);
            }
        }
    }

    return {Tensor({finalW, finalH, tensorShape[2]}, data), argmax};
}
