// POSIX imports
#include <algorithm>
#include <arpa/inet.h>
#include <cstdint>
#include <errno.h>
#include <iostream>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

// CNN-related
#include "mnist.h"
#include "network.h"
#include "tensor.h"
#include <vector>

// WebSocket-related
#include "base64.h"
#include "sha-1.h"

#define PORT 8080 // The port users will connect to
#define BUFFER_SIZE 1024

// Reads exactly n bytes (appending to out). Returns false if peer closed / error.
bool readExactly(int fd, std::vector<unsigned char>& out, size_t n) {
    size_t have = 0;
    unsigned char tmp[BUFFER_SIZE];
    while (have < n) {
        ssize_t r = read(fd, tmp, std::min(n - have, (size_t)BUFFER_SIZE));
        if (r <= 0)
            return false; // 0 = closed, <0 = error
        out.insert(out.end(), tmp, tmp + r);
        have += r;
    }
    return true;
}

int main() {
    char buffer[BUFFER_SIZE];

    // ─── Build the socket ────────────────────────────────────────────────────────
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("webserver (socket)");
        return 1;
    }
    std::cout << "Socket created successfully\n";

    // Allow re-binding to the same address (avoids "Address already in use" during TIME_WAIT after restart)
    int opt = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0) {
        perror("webserver (setsockopt)");
        return 1;
    }

    // ─── Bind an address to the socket ───────────────────────────────────────────
    // Create the address to bind the socket to (i.e. addrinfo)
    struct sockaddr_in host_addr;
    int host_addrlen = sizeof(host_addr);

    host_addr.sin_family = AF_INET;
    host_addr.sin_port = htons(PORT);
    host_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    // (VERBOSE) Create client address
    struct sockaddr_in client_addr;
    int client_addrlen = sizeof(client_addr);

    if (bind(sockfd, (struct sockaddr*)&host_addr, host_addrlen) != 0) {
        perror("webserver (bind)");
        return 1;
    }
    std::cout << "socket successfully bound to address\n";

    // ─── Listen for incoming connections ─────────────────────────────────────────
    if (listen(sockfd, SOMAXCONN) != 0) {
        perror("webserver (listen)");
        return 1;
    }
    std::cout << "server listening for connections\n";

    // ─── Initialize the CNN model ────────────────────────────────────────────────
    Network net;           // Builds the fixed architecture
    net.load("model.bin"); // Loads the trained model
    std::cout << "model is loaded\n";

    // Load MNIST dataset to verify prediction works
    std::vector<Tensor> testImages = mnist::loadImages("../data/t10k-images.idx3-ubyte");
    std::vector<int> testLabels = mnist::loadLabels("../data/t10k-labels.idx1-ubyte");
    std::cout << "images and labels are loaded\n";

    // Grab first image to test with
    Tensor testImg = testImages[0];
    std::cout << "test image actual label: " << testLabels[0] << "\n";

    // ─── Accept incoming connections ─────────────────────────────────────────────
    while (true) {
        int newsockfd = accept(sockfd, (struct sockaddr*)&host_addr, (socklen_t*)&host_addrlen);
        if (newsockfd < 0) {
            perror("webserver (accept)");
            continue;
        }
        std::cout << "connection accepted\n";

        // (VERBOSE) Get client address
        int sockn = getsockname(newsockfd, (struct sockaddr*)&client_addr, (socklen_t*)&client_addrlen);
        if (sockn < 0) {
            perror("webserver (getsockname)");
            continue;
        }

        // ─── Read the HTTP handshake request ─────────────────────────────────────
        // The handshake is a small text request, so one read is fine here.
        int valread = read(newsockfd, buffer, BUFFER_SIZE);
        if (valread < 0) {
            perror("webserver (read)");
            continue;
        }

        std::cout << "[" << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << "]\n";
        std::cout << "─── Raw request ───\n";
        std::cout.write(buffer, valread);
        std::cout << "\n───────────────────\n";

        std::string response;

        // Convert buffer bytes into string
        std::string headers(buffer, valread);

        size_t upgrade = headers.find("Upgrade: websocket");
        if (upgrade != std::string::npos) {
            // Extract key
            size_t pos = headers.find("Sec-WebSocket-Key:");
            size_t valueStart = pos + strlen("Sec-WebSocket-Key: ");
            size_t valueEnd = headers.find("\r\n", valueStart);
            std::string key = headers.substr(valueStart, valueEnd - valueStart);
            std::cout << "[" << key << "]\n";

            std::string combined = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
            std::string accept = hash(combined.c_str());

            response =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " +
                accept + "\r\n"
                         "\r\n";

            int valwrite = write(newsockfd, response.c_str(), response.size());
            if (valwrite < 0) {
                perror("webserver (write)");
                continue;
            }

            // ─── Frame loop ──────────────────────────────────────────────────────
            while (true) {
                std::vector<unsigned char> rawFrame; // accumulated for the hex dump

                // 1. First 2 bytes: FIN/opcode + MASK/length-indicator. hdr -> "header"
                std::vector<unsigned char> hdr;
                if (!readExactly(newsockfd, hdr, 2)) {
                    std::cout << "connection closed\n";
                    break;
                }
                rawFrame.insert(rawFrame.end(), hdr.begin(), hdr.end());

                bool fin = hdr[0] & 0x80; // not used yet (see note on fragmentation)
                unsigned char opcode = hdr[0] & 0x0F;
                bool masked = hdr[1] & 0x80;
                uint64_t len = hdr[1] & 0x7F;

                // 2. Extended payload length
                if (len == 126) {
                    std::vector<unsigned char> ext;
                    if (!readExactly(newsockfd, ext, 2))
                        break;
                    rawFrame.insert(rawFrame.end(), ext.begin(), ext.end());
                    len = ((uint64_t)ext[0] << 8) | ext[1]; // 16-bit big-endian
                } else if (len == 127) {
                    std::vector<unsigned char> ext;
                    if (!readExactly(newsockfd, ext, 8))
                        break;
                    rawFrame.insert(rawFrame.end(), ext.begin(), ext.end());
                    len = 0;
                    for (int i = 0; i < 8; ++i)
                        len = (len << 8) | ext[i]; // 64-bit big-endian
                }

                // 3. Mask key (client→server frames are always masked)
                unsigned char maskKey[4] = {0, 0, 0, 0};
                if (masked) {
                    std::vector<unsigned char> mk;
                    if (!readExactly(newsockfd, mk, 4))
                        break;
                    rawFrame.insert(rawFrame.end(), mk.begin(), mk.end());
                    for (int i = 0; i < 4; ++i)
                        maskKey[i] = mk[i];
                }

                // 4. Payload
                std::vector<unsigned char> payload;
                if (!readExactly(newsockfd, payload, len))
                    break;
                rawFrame.insert(rawFrame.end(), payload.begin(), payload.end());
                for (uint64_t i = 0; i < len; ++i)
                    payload[i] ^= maskKey[i % 4];

                // (VERBOSE) Frame summary + hex dump
                std::cout << "─── Frame (" << rawFrame.size() << " bytes, opcode 0x"
                          << std::hex << (int)opcode << std::dec
                          << ", payload " << len << ") ───\n";
                for (size_t i = 0; i < rawFrame.size(); ++i) {
                    printf("%02X ", rawFrame[i]);
                }
                printf("\n");

                // ─── Payload → normalized Tensor → prediction ────────────────
                // The payload is 784 raw bytes (0–255), one per pixel, row-major.
                if (opcode == 0x2 && len == 784) {
                    // Normalize exactly as mnist::loadImages does: byte / 255.0f
                    std::vector<float> pixels(784);
                    for (int j = 0; j < 784; ++j) {
                        pixels[j] = payload[j] / 255.0f;
                    }

                    // Same shape mnist::loadImages builds: {cols, rows, 1} = {28, 28, 1}
                    Tensor input({28, 28, 1}, pixels);

                    // Predict — Network::predict returns the highest-probability class index
                    Tensor probs = net.predictProbabilities(input); // {10, 1, 1}, softmax
                    const std::vector<float>& p = probs.getData();  // 10 floats, row-major

                    // Serialize as text: "0.01,0.00,0.95,..." (10 comma-separated values)
                    std::string out;
                    for (size_t k = 0; k < p.size(); ++k) {
                        if (k)
                            out += ",";
                        out += std::to_string(p[k]);
                    }

                    // Text frame. out is well under 125 bytes (10 values × ~8 chars ≈ 80),
                    // so the direct-length branch applies.
                    std::vector<unsigned char> frame;
                    frame.push_back(0x81); // FIN + text opcode
                    frame.push_back((unsigned char)out.size());
                    frame.insert(frame.end(), out.begin(), out.end());

                    int valWrite = write(newsockfd, frame.data(), frame.size());
                    if (valWrite < 0) {
                        perror("webserver (prediction write)");
                        break;
                    }
                } else {
                    // Non-784 or non-binary frame (e.g. a close/ping/text frame): ignore for now.
                    std::cout << "ignoring frame (opcode 0x" << std::hex << (int)opcode
                              << std::dec << ", payload " << len << ")\n";
                }
            }

        } else {
            response =
                "HTTP/1.0 200 OK\r\n"
                "Server: webserver-cpp\r\n"
                "Content-type: text/html\r\n\r\n"
                "<html>Hello World<html>\r\n";

            int valwrite = write(newsockfd, response.c_str(), response.size());
            if (valwrite < 0) {
                perror("webserver (write)");
                continue;
            }
        }

        close(newsockfd);
    }

    return 0;
}