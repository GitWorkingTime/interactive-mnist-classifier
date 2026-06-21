#ifndef BASE_LAYER_H
#define BASE_LAYER_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"

// ─── Class declarations ──────────────────────────────────────────────────────
class BaseLayer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradInput) = 0;
    virtual ~BaseLayer() = default;
};

#endif
