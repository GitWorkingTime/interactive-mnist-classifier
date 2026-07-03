# Interactive MNIST Classifier

A real-time handwritten-digit recognition app. A browser canvas streams a drawing over a
WebSocket to a from-scratch C++ server, which runs a convolutional neural network (CNN) and
returns a live probability distribution over the digits 0–9 as you draw.

The C++ side implements the WebSocket protocol, the CNN, and the tensor math by hand. No
web-server or deep-learning frameworks.

---

## Contents

- [Architecture](#architecture)
- [Data flow](#data-flow)
- [Server](#server)
- [Model](#model)
- [Frontend](#frontend)
- [Preprocessing (important)](#preprocessing-important)
- [Build & run](#build--run)
- [Known limitations](#known-limitations)
- [Resources](#resources)
- [Provenance of this README](#provenance-of-this-readme)

---

## Architecture

```
┌────────────────────┐        ws://localhost:8080         ┌───────────────────────────┐
│   React frontend   │  ───────────────────────────────►  │      C++ WebSocket server │
│                    │      784 raw bytes (binary frame)  │                           │
│  • canvas (280²)   │                                    │  • hand-rolled WS framing │
│  • debug log       │                                    │  • normalize /255         │
│  • confidence bars │  ◄───────────────────────────────  │  • CNN forward pass       │
└────────────────────┘   10 probabilities (text frame)    │  • softmax → 10 probs     │
                                                          └───────────────────────────┘
```

The frontend is split into three columns: a debug log (left), the drawing canvas (middle),
and a horizontal confidence bar graph (right).

---

## Data flow

1. The user draws on a 280×280 canvas (white stroke on black background).
2. On each change (throttled to at most once per 100 ms) the drawing is cropped to its
   bounding box, scaled to fit a 20×20 box preserving aspect ratio, centered in a 28×28
   field, and reduced to 784 grayscale bytes (0–255).
3. Those 784 bytes are sent as a single binary WebSocket frame.
4. The server parses and unmasks the frame, divides each byte by 255 to get values in
   [0, 1], and builds a `{28, 28, 1}` tensor.
5. The tensor is run through the CNN; a softmax produces 10 class probabilities.
6. The server sends the 10 probabilities back as a comma-separated text frame.
7. The frontend parses them, updates the bar graph, and appends them to the debug log.

The payload is 784 bytes because a 28×28 single-channel image has 28 × 28 = 784 pixels.

---

## Server

A single-threaded C++ server built directly on POSIX sockets. It:

- Creates, binds, and listens on a TCP socket on port `8080` (`SO_REUSEADDR` is set so the
  server can be restarted immediately without waiting out `TIME_WAIT`).
- Performs the WebSocket opening handshake: it reads the HTTP upgrade request, extracts the
  `Sec-WebSocket-Key`, concatenates the RFC-defined GUID
  `258EAFA5-E914-47DA-95CA-C5AB0DC85B11`, hashes the result with SHA-1, Base64-encodes it,
  and returns it as `Sec-WebSocket-Accept` in a `101 Switching Protocols` response.
- Parses inbound data frames by hand, including the extended-length encoding: the 7-bit
  length field carries lengths 0–125 directly, the value 126 signals a 16-bit big-endian
  length, and 127 signals a 64-bit big-endian length. Client→server frames are always
  masked, so the payload is unmasked with the 4-byte masking key.
- Uses a `readExactly` helper that loops on `read()` until the requested number of bytes has
  arrived, because a single `read()` on a TCP stream may return fewer bytes than requested.

A 784-byte payload uses the 16-bit (value 126) length branch, since 784 falls between 126
and 65535.

### Non-WebSocket requests

A plain HTTP request (no `Upgrade: websocket` header) receives a minimal `200 OK`
`Hello World` response.

---

## Model

A fixed-architecture CNN, as documented in the project's `network.h`:

```
input        {28, 28, 1}
Conv 3x3     {26, 26, 8}
ReLU         {26, 26, 8}
MaxPool 2x2  {13, 13, 8}
FC: 13*13*8 = 1352 -> 10 classes
```

Trained parameters are loaded from `model.bin` at startup via `Network::load`.

The model exposes three inference entry points (per `network.h`): `forward` (raw scores),
`predictProbabilities` (a `{10, 1, 1}` softmax distribution), and `predict` (the argmax class
index). The live app uses `predictProbabilities` so it can display the full distribution.

### Tensor library

All model math is backed by a custom `Tensor` class (`tensor.h`) storing data in a flat
row-major float array with shape `{W, H, D}`. It provides the operations the CNN needs,
including `convolve`, `multiply` (matrix multiply), `transpose`, `flatten`, `pad`,
`rotate180`, `getSlice`/`stackSlices`, and element-wise operations.

### Data loading

MNIST data is read from IDX files (`mnist.h` / `mnist.cpp`). The loader validates the IDX
magic numbers (2051 for images, 2049 for labels), byte-swaps the big-endian header integers,
and normalizes each pixel to [0, 1] by dividing by 255. Each image becomes a `{28, 28, 1}`
tensor.

### Reported training results

The following figures were reported by the project author and are **not independently
verified in this README**: the model was trained for 3 epochs over the 60,000 MNIST training
images, reaching ~98% training accuracy where accuracy plateaued (rising from ~75% to ~98%
with a learning rate of 0.01 and no observed gradient divergence), and ~97% on a separate
test set. These are the author's stated results.

---

## Frontend

A single-file React component (`App.jsx`) styled with Tailwind utility classes. It:

- Opens a WebSocket to `ws://localhost:8080` and reports connection status with a colored
  indicator.
- Renders a 280×280 canvas with pointer-based drawing (black background, white pen).
- Sends the preprocessed 784-byte payload on each drawing change, throttled to at most once
  every 100 ms, plus once more when a stroke ends.
- Parses the returned comma-separated probabilities and renders a horizontal bar per digit
  (0–9) on a fixed 0–100% axis, highlighting the highest-probability digit in green.
- Keeps a timestamped debug log (connection events, bytes sent, and the predicted
  probabilities) that auto-scrolls to the newest entry.

---

## Preprocessing

MNIST training digits are size-normalized and centered within the 28×28 field. A plain
downscale of the full canvas does **not** reproduce that, and during development this caused
confident misclassifications (e.g. a drawn 9 predicted as 4).

The frontend therefore preprocesses each drawing before sending it:

1. Scan the canvas for non-background pixels and compute their bounding box.
2. Scale the cropped digit so its longer side becomes 20 px, preserving aspect ratio.
3. Center it in a 28×28 field.
4. Extract 784 grayscale bytes.

This bounding-box approach centers by **geometry**, not by center of mass. Canonical MNIST
centers by center of mass; the difference between the two is a known remaining approximation
in this project and is **not confirmed** to be negligible.

Server-side normalization (`byte / 255`) exactly matches the training-time normalization in
`mnist.cpp`, so canvas input and training input use the same [0, 1] scaling.

---

## Build & run

> Note: exact build commands are **not specified** in the materials this README was written
> from. The steps below describe what the code requires; adapt them to your build setup.

The server requires:

- A C++ compiler with C++11 or later (the code uses `<cstdint>`, `std::vector`, etc.).
- POSIX socket headers (`<sys/socket.h>`, `<arpa/inet.h>`, `<unistd.h>`) so Linux or
  macOS. If on windows, use WSL2.
- The trained model file `model.bin` in the working directory. This can be done by going into `build/` and running the follow commands:
```
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 16
./training
```
- The MNIST dataset. These can be found on kaggle via this [page](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data). Make sure to copy the files under `data/`

The frontend requires a React toolchain with Tailwind CSS configured.

Typical flow:

1. Build and start the C++ server; it listens on port `8080`.
2. Start the React dev server and open the app in a browser.
3. Draw a digit; predictions update live.

---

## Known limitations

These are open items identified during development:

- **Control frames are not handled.** Close (`0x8`), ping (`0x9`), and pong (`0xA`) frames
  are not processed; when the browser tab closes, the close frame is ignored rather than
  answered with a clean close.
- **Fragmented frames are not reassembled.** The FIN bit is parsed but not acted on. For
  784-byte payloads a browser will typically send a single unfragmented frame, but this is
  **not guaranteed** for all browsers or payload sizes.
- **Single-threaded, one connection at a time.** The server handles connections sequentially.
- **Geometric vs center-of-mass centering.** See [Preprocessing](#preprocessing-important).
- **Occasional misclassification.** Some digits misclassify.

---

## Resources

Deep learning & math
- Stanford's CS231n: Deep Learning for Computer Vision [link](https://cs231n.github.io/)
- University of Toronto CSC321 — Lecture 6: Backpropagation (Roger Grosse) [link](https://www.cs.toronto.edu/~lczhang/321/notes/notes06.pdf)
- Introduction to CNN's [link](https://jasoncantarella.com/downloads/CNN.pdf)
- The Matrix Cookbook [link](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
- 3Blue1Brown Neural Network Playlist [link](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- 3Blue1Brown Essence of Linear Algebra [link](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- 3Blue1Brown Essence of Calculus [link](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

Networking & protocols
- Beej's Guide to Network Programming [link](https://beej.us/guide/bgnet/html/)
- brainsslot.jp's Making a simple HTTP webserver in C [link](https://bruinsslot.jp/post/simple-http-webserver-in-c/)
- MDN's Writing WebSocket server [link](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_servers)
- RFC 3174: US Secure Hashing Algorithm (SHA-1) [link](https://www.rfc-editor.org/rfc/rfc3174.html)
- sha1-explained [link](https://github.com/thomas-barthelemy/sha1-explained)
- RFC 4648: The Base16, Base32, Base64 Encoding [link](https://www.rfc-editor.org/rfc/rfc4648)
- RFC 6455: The WebSocket protocol [link](https://www.rfc-editor.org/rfc/rfc6455)

---