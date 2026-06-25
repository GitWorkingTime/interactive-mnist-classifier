#include "mnist.h"

// Parsing is based on this documentation:
// https://github.com/sunsided/mnist

int mnist::reverseInt(int num) {
    unsigned char b1, b2, b3, b4;
    b1 = num & 255;
    b2 = (num >> 8) & 255;
    b3 = (num >> 16) & 255;
    b4 = (num >> 24) & 255;
    return ((int)b1 << 24) | ((int)b2 << 16) | ((int)b3 << 8) | ((int)b4);
}

std::vector<Tensor> mnist::loadImages(const std::string& path, bool verbose) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // The image header is 16 bytes (4 big-endian ints). Reject anything smaller
    // before we read past the end of the buffer.
    if (fileSize < 16)
        throw std::runtime_error("File too small to be a valid IDX image file: " + path);

    std::vector<unsigned char> buffer(fileSize);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    int magic = reverseInt(*(int*)(buffer.data() + 0));
    int numImages = reverseInt(*(int*)(buffer.data() + 4));
    int rows = reverseInt(*(int*)(buffer.data() + 8));
    int cols = reverseInt(*(int*)(buffer.data() + 12));

    if (magic != 2051)
        throw std::runtime_error("Invalid magic number");

    // Verify the file actually holds the pixel data it claims to.
    std::streamsize expected = 16 + static_cast<std::streamsize>(numImages) * rows * cols;
    if (fileSize < expected)
        throw std::runtime_error("File truncated: header claims more image data than the file contains: " + path);

    if (verbose) {
        std::cout << " ── Image Metadata ───────────" << std::endl;
        std::cout << "Magic Number: " << magic << std::endl;
        std::cout << "Number of Images: " << numImages << std::endl;
        std::cout << "Number of Rows: " << rows << std::endl;
        std::cout << "Number of Columns: " << cols << std::endl;
        std::cout << " ─────────────────────────────" << std::endl;
    }

    std::vector<Tensor> images;
    images.reserve(numImages);

    int imageSize = rows * cols;
    for (int i = 0; i < numImages; ++i) {
        std::vector<float> pixels(imageSize);
        int offset = 16 + i * imageSize;
        for (int j = 0; j < imageSize; ++j) {
            pixels[j] = buffer[offset + j] / 255.0f;
        }
        images.push_back(Tensor({cols, rows, 1}, pixels));
    }

    return images;
}

std::vector<int> mnist::loadLabels(const std::string& path, bool verbose) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Error: Couldn't open file: " + path);

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // The label header is 8 bytes (2 big-endian ints).
    if (fileSize < 8)
        throw std::runtime_error("File too small to be a valid IDX label file: " + path);

    std::vector<unsigned char> buffer(fileSize);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    int magic = reverseInt(*(int*)(buffer.data() + 0));
    int numLabels = reverseInt(*(int*)(buffer.data() + 4));

    if (magic != 2049)
        throw std::runtime_error("Invalid magic number");

    std::streamsize expected = 8 + static_cast<std::streamsize>(numLabels);
    if (fileSize < expected)
        throw std::runtime_error("File truncated: header claims more labels than the file contains: " + path);

    if (verbose) {
        std::cout << " ── Label Metadata ───────────" << std::endl;
        std::cout << "Magic Number: " << magic << std::endl;
        std::cout << "Number of Labels: " << numLabels << std::endl;
        std::cout << " ─────────────────────────────" << std::endl;
    }

    std::vector<int> labels;
    labels.reserve(numLabels);

    int offset = 8;
    for (int i = 0; i < numLabels; ++i) {
        labels.push_back(buffer[offset + i]);
    }

    return labels;
}

Tensor mnist::oneHotEncodeLabels(int label) {
    if (label < 0 || label > 9) {
        throw std::invalid_argument(
            "ERROR: Label must be between 0 and 9 - received: " + std::to_string(label));
    }

    std::vector<float> encode(10, 0.0f); // ten zeros
    encode[label] = 1.0f;                // set the one hot position
    return Tensor({1, 10, 1}, encode);
}