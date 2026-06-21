#include "mnist.h"
#include "tensor.h"
#include <iostream>
#include <numeric>

int main() {
    std::vector<Tensor> images = mnist::loadImages("../data/train-images.idx3-ubyte");
    std::vector<int> labels = mnist::loadLabels("../data/train-labels.idx1-ubyte");

    images[0].displayASCII();
    std::cout << labels[0] << std::endl;

    Tensor oneHot = mnist::oneHotEncodeLabels(labels[0]);
    oneHot.displayRaw();

    return 0;
}