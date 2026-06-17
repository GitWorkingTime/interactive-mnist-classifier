cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# ── Main Executable ───────────────────────────────────────────────────────────
add_executable(main
    # ── Add Files here ────────────────────────────────────────────────────────
    "src/main.cpp"
    "src/cnn/tensor.cpp"
)

# ── Header Files ──────────────────────────────────────────────────────────────
target_include_directories(main PRIVATE
${PROJECT_SOURCE_DIR}/includes/cnn
${CMAKE_BINARY_DIR}/generated
)

# ── Testing ───────────────────────────────────────────────────────────────────
function(add_cnn_test name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE
        ${PROJECT_SOURCE_DIR}/includes/cnn
        ${CMAKE_BINARY_DIR}/generated
    )
endfunction()

add_cnn_test(test_tensor
    "tests/cnn/test_tensor.cpp"
    "src/cnn/tensor.cpp"
)

add_cnn_test(test_activations
    "tests/cnn/test_activations.cpp"
    "src/cnn/activations.cpp"
    "src/cnn/tensor.cpp"
)