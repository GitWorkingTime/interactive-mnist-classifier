// POSIX imports
#include <arpa/inet.h>
#include <errno.h>
#include <iostream>
#include <sys/socket.h>

#define PORT 8080 // The port users will connect to

int main() {
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

    if (bind(sockfd, (struct sockaddr*)&host_addr, host_addrlen) != 0) {
        perror("webserver (bind)");
        return 1;
    }
    std::cout << "socket successfully bound to address\n";

    // ─── Listen to the socket ────────────────────────────────────────────────────
    // ─── Accept the socket (?) ───────────────────────────────────────────────────
    // ─── Allow read/write on the socket ──────────────────────────────────────────

    return 0;
}