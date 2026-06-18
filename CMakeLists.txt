cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ── Main Executable ───────────────────────────────────────────────────────────
add_executable(main
    # ── Add Files here ────────────────────────────────────────────────────────
    "src/main.cpp"
    "src/cnn/operations/tensor.cpp"
    "src/cnn/operations/activations.cpp"
    "src/cnn/operations/pooling.cpp"
)

# ── Header Files ──────────────────────────────────────────────────────────────
target_include_directories(main PRIVATE
${PROJECT_SOURCE_DIR}/includes/cnn/operations
${CMAKE_BINARY_DIR}/generated
)

# ── Testing ───────────────────────────────────────────────────────────────────
function(add_cnn_test name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE
        ${PROJECT_SOURCE_DIR}/includes/cnn/operations
        ${CMAKE_BINARY_DIR}/generated
    )
endfunction()

add_cnn_test(test_tensor
    "tests/cnn/test_tensor.cpp"
    "src/cnn/operations/tensor.cpp"
)

add_cnn_test(test_activations
    "tests/cnn/test_activations.cpp"
    "src/cnn/operations/activations.cpp"
    "src/cnn/operations/tensor.cpp"
)

add_cnn_test(test_pooling
    "tests/cnn/test_pooling.cpp"
    "src/cnn/operations/pooling.cpp"
    "src/cnn/operations/tensor.cpp"
)

add_cnn_test(test_loss
    "tests/cnn/test_loss.cpp"
    "src/cnn/operations/loss.cpp"
    "src/cnn/operations/tensor.cpp"
)