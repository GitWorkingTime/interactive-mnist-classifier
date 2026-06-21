#ifndef RELU_LAYER_H
#define RELU_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "activations.h"
#include "base_layer.h"
#include "tensor.h"

// ─── Class declarations ──────────────────────────────────────────────────────
class ReLULayer : public BaseLayer {

private:
    // Stashed during forward, needed for backward
    Tensor input;

public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
};

#endif