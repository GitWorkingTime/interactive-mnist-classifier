// POSIX imports
#include <arpa/inet.h>
#include <errno.h>
#include <iostream>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 8080 // The port users will connect to
#define BUFFER_SIZE 1024

int main() {
    char buffer[BUFFER_SIZE];
    char resp[] = "HTTP/1.0 200 OK\r\n"
                  "Server: webserver-cpp\r\n"
                  "Content-type: text/html\r\n\r\n"
                  "<html>Hello World</html>\r\n";

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

        // (VERBOSE) Log client information
        std::cout << "[" << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << "]\n";

        // (VERBOSE) Read the request
        char method[BUFFER_SIZE], uri[BUFFER_SIZE], version[BUFFER_SIZE];
        sscanf(buffer, "%s %s %s", method, uri, version);

        std::cout << "method: " << method << "\n";
        std::cout << "uri: " << uri << "\n";
        std::cout << "version: " << version << "\n";

        // ─── Write to the socket ─────────────────────────────────────────────────
        int valwrite = write(newsockfd, resp, strlen(resp));
        if (valwrite < 0) {
            perror("webserver (write)");
            continue;
        }

        close(newsockfd);
    }

    return 0;
}