// POSIX imports
#include <arpa/inet.h>
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

int main() {
    char buffer[BUFFER_SIZE];

    // ─── Build the socket ────────────────────────────────────────────────────────
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        perror("webserver (socket)");
        return 1;
    }
    std::cout << "Socket created successfully\n";

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
        // Accept incoming connections
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

        // ─── Read from the socket ────────────────────────────────────────────────
        int valread = read(newsockfd, buffer, BUFFER_SIZE);
        if (valread < 0) {
            perror("webserver (read)");
            continue;
        }

        // Make prediction
        int predicted = net.predict(testImg);
        std::cout << "predicted digit: " << predicted << "\n";

        // (VERBOSE) Log client information
        std::cout << "[" << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << "]\n";

        // (VERBOSE) Read the request
        std::cout << "─── Raw request ───\n";
        std::cout.write(buffer, valread);
        std::cout << "\n───────────────────\n";

        std::string response;

        // Parse the request header to find the Sec-WebSocket-Key
        std::string headers(buffer, valread);

        // Verify that the request is upgrading to websocket:
        size_t upgrade = headers.find("Upgrade: websocket");
        if (upgrade != std::string::npos) {
            // Extract key
            size_t pos = headers.find("Sec-WebSocket-Key:");
            size_t valueStart = pos + strlen("Sec-WebSocket-Key: ");
            size_t valueEnd = headers.find("\r\n", valueStart);
            std::string key = headers.substr(valueStart, valueEnd - valueStart);
            std::cout << "[" << key << "]\n";

            // Pass it through sha-1
            std::string combined = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
            std::string accept = hash(combined.c_str());

            // Create response
            response =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " +
                accept + "\r\n"
                         "\r\n";
        } else {
            // Make the response body
            std::string body = "<html>Predicted digit: " + std::to_string(predicted) + "</html>\r\n";
            response =
                "HTTP/1.0 200 OK\r\n"
                "Server: webserver-cpp\r\n"
                "Content-type: text/html\r\n\r\n" +
                body;
        }

        // ─── Write to the socket ─────────────────────────────────────────────────

        int valwrite = write(newsockfd, response.c_str(), response.size());
        if (valwrite < 0) {
            perror("webserver (write)");
            continue;
        }

        // close(newsockfd);
    }

    return 0;
}