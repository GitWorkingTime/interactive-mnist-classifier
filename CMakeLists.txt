cmake_minimum_required(VERSION 3.10)
project(InteractiveMNISTClassifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Include directories (shared by all modules) ───────────────────────────────
# Each library exposes these PUBLICLY so anything linking the library inherits
# the include paths automatically.
set(CNN_INCLUDES
    ${PROJECT_SOURCE_DIR}/includes/cnn/operations
    ${PROJECT_SOURCE_DIR}/includes/cnn/helpers
    ${PROJECT_SOURCE_DIR}/includes/cnn/layers
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

# ── Main Executable ───────────────────────────────────────────────────────────
add_executable(main "src/main.cpp")
target_link_libraries(main PRIVATE
    tensor
    activations
    pooling
    loss
    mnist
    weights_init
    relu_layer
    maxpool_layer
    fc_layer
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
