cmake_minimum_required(VERSION 3.10)
project(InteractiveMNISTClassifier)

# Default to a Release build (optimized) if the user didn't specify
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Include directories (shared by all modules) ───────────────────────────────
# Each library exposes these PUBLICLY so anything linking the library inherits
# the include paths automatically.
set(CNN_INCLUDES
    ${PROJECT_SOURCE_DIR}/includes/cnn/operations
    ${PROJECT_SOURCE_DIR}/includes/cnn/helpers
    ${PROJECT_SOURCE_DIR}/includes/cnn/layers
    ${PROJECT_SOURCE_DIR}/includes/cnn/
    ${CMAKE_BINARY_DIR}/generated
)

# ── Operation libraries ───────────────────────────────────────────────────────
# tensor is the foundation — everything else depends on it.
add_library(tensor "src/cnn/operations/tensor.cpp")
target_include_directories(tensor PUBLIC ${CNN_INCLUDES})

add_library(activations "src/cnn/operations/activations.cpp")
target_include_directories(activations PUBLIC ${CNN_INCLUDES})
target_link_libraries(activations PUBLIC tensor)        # activations uses Tensor

add_library(pooling "src/cnn/operations/pooling.cpp")
target_include_directories(pooling PUBLIC ${CNN_INCLUDES})
target_link_libraries(pooling PUBLIC tensor)            # pooling uses Tensor

add_library(loss "src/cnn/operations/loss.cpp")
target_include_directories(loss PUBLIC ${CNN_INCLUDES})
target_link_libraries(loss PUBLIC tensor)               # loss uses Tensor

# ── Helper libraries ──────────────────────────────────────────────────────────
add_library(mnist "src/cnn/helpers/mnist.cpp")
target_include_directories(mnist PUBLIC ${CNN_INCLUDES})
target_link_libraries(mnist PUBLIC tensor)              # mnist builds Tensors

add_library(weights_init "src/cnn/helpers/weights_init.cpp")
target_include_directories(weights_init PUBLIC ${CNN_INCLUDES})
target_link_libraries(weights_init PUBLIC tensor)       # weights_init builds Tensors

# ── Layer libraries ───────────────────────────────────────────────────────────
add_library(relu_layer "src/cnn/layers/ReLU_layer.cpp")
target_include_directories(relu_layer PUBLIC ${CNN_INCLUDES})
target_link_libraries(relu_layer PUBLIC activations tensor)  # ReLU layer uses both

add_library(maxpool_layer "src/cnn/layers/maxPooling_layer.cpp")
target_include_directories(maxpool_layer PUBLIC ${CNN_INCLUDES})
target_link_libraries(maxpool_layer PUBLIC pooling tensor)

add_library(fc_layer "src/cnn/layers/fc_layer.cpp")
target_include_directories(fc_layer PUBLIC ${CNN_INCLUDES})
target_link_libraries(fc_layer PUBLIC weights_init tensor)

add_library(conv_layer "src/cnn/layers/conv_layer.cpp")
target_include_directories(conv_layer PUBLIC ${CNN_INCLUDES})
target_link_libraries(conv_layer PUBLIC weights_init tensor)

add_library(network "src/cnn/network.cpp")
target_include_directories(network PUBLIC ${CNN_INCLUDES})
target_link_libraries(network PUBLIC
    tensor
    activations
    loss
    relu_layer
    maxpool_layer
    fc_layer
    conv_layer
)

# ── Web app include directories ───────────────────────────────────────────────
set(WEB_APP_INCLUDES
    ${PROJECT_SOURCE_DIR}/includes/web_app/encode
    ${PROJECT_SOURCE_DIR}/includes/web_app/hash
)

# ── Web app libraries ─────────────────────────────────────────────────────────
add_library(base64 "src/web_app/backend/encode/base64.cpp")
target_include_directories(base64 PUBLIC ${WEB_APP_INCLUDES})

add_library(sha1 "src/web_app/backend/hash/sha-1.cpp")
target_include_directories(sha1 PUBLIC ${WEB_APP_INCLUDES})
target_link_libraries(sha1 PUBLIC base64)

# ── Main Executable ───────────────────────────────────────────────────────────
add_executable(training "src/cnn/training.cpp")
target_link_libraries(training PRIVATE
    network
    mnist
)

add_executable(main "src/cnn/main.cpp")
target_link_libraries(main PRIVATE
    network
    mnist
)

add_executable(server "src/web_app/backend/server.cpp")
target_link_libraries(server PRIVATE
    network
    mnist
    base64
    sha1
)

# ── Testing ───────────────────────────────────────────────────────────────────
# Each test links ONLY the libraries it directly needs; CMake pulls in the
# transitive dependencies (e.g. linking relu_layer also brings activations + tensor).
enable_testing()

function(add_cnn_test name testfile)
    add_executable(${name} ${testfile})
    target_link_libraries(${name} PRIVATE ${ARGN})  # ARGN = the libraries to link
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_cnn_test(test_tensor      "tests/cnn/operations/test_tensor.cpp"       tensor)
add_cnn_test(test_activations "tests/cnn/operations/test_activations.cpp"  activations)
add_cnn_test(test_pooling     "tests/cnn/operations/test_pooling.cpp"      pooling)
add_cnn_test(test_loss        "tests/cnn/operations/test_loss.cpp"         loss)
add_cnn_test(test_weights     "tests/cnn/helpers/test_weights.cpp"         weights_init)
add_cnn_test(test_mnist       "tests/cnn/helpers/test_mnist.cpp"           mnist)
add_cnn_test(test_relu        "tests/cnn/layers/test_relu.cpp"             relu_layer)
add_cnn_test(test_maxpool     "tests/cnn/layers/test_maxpool.cpp"          maxpool_layer)
add_cnn_test(test_fc          "tests/cnn/layers/test_fc.cpp"               fc_layer)
add_cnn_test(test_conv        "tests/cnn/layers/test_conv.cpp"             conv_layer)
add_cnn_test(test_base64      "tests/web_app/test_base64.cpp"              base64)
add_cnn_test(test_sha1        "tests/web_app/test_sha1.cpp"                sha1)


