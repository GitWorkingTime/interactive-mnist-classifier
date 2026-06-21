// ─── Imports ─────────────────────────────────────────────────────────────────
#include "weights_init.h"

// ─── Function Defintions ─────────────────────────────────────────────────────
Tensor weights::heInit(const std::vector<int>& shape, int fanIn) {
    // One generator, created once, reused across calls so each call advances
    // the sequence (otherwise every filter would get identical values).
    static std::mt19937 gen(std::random_device{}());

    float stddev = std::sqrt(2.0f / fanIn);
    std::normal_distribution<float> dist(0.0f, stddev);

    int count = shape[0] * shape[1] * shape[2];
    std::vector<float> data = {};
    data.reserve(count);
    for (int i = 0; i < count; ++i) {
        data.push_back(dist(gen));
    }

    return Tensor(shape, data);
}