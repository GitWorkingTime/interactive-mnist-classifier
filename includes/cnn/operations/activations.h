#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// ─── Imports ─────────────────────────────────────────────────────────────────
#include "tensor.h"

// ─── Activations ─────────────────────────────────────────────────────────────
namespace activations {
/**
 * @brief performs a ReLU operation on a given tensor
 *
 * @par Description
 * Given a tensor, it returns the original tensor but
 * for every negative value, it's replaced with 0
 *
 * ReLU, mathematically, is a piecewise function:
 * x <= 0: f(x) = 0
 * x > 0: f(x) = x
 *
 * @param tensor The tensor to perform the ReLU operation on
 *
 * @par Example
 * @code
 * Tensor a({2, 2, 1}, {-1, 2, 3, -4});
 * Tensor b = activations::ReLU(a); // Data: {0, 2, 3, 0};
 * @endcode
 */
Tensor ReLU(const Tensor& tensor);

/**
 * @brief performs a ReLU derivative operation on a given tensor
 *
 * @par Description
 * Given a tensor, it returns a tensor where every positive integer
 * is replaced with a 1.0 and every negative tensor is replaced with
 * a 0
 *
 * The derivative of ReLU, mathematically, is this piecewise function:
 * x <= 0: f'(x) = 0
 * x > 0: f'(x) = 1
 *
 * @param tensor The tensor to perform the ReLU derivative operation on
 *
 * @par Example
 * @code
 * Tensor a({2, 2, 1}, {-1, 2, 3, -4});
 * Tensor b = activations::ReLUDerivative(a); // Data: {0, 1, 1, 0}
 * @endcode
 */
Tensor ReLUDerivative(const Tensor& tensor);
} // namespace activations

#endif