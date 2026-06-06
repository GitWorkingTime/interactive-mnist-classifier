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
add_executable(test
    # ── Add Files here ────────────────────────────────────────────────────────
    "tests/cnn/test_tensor.cpp"
    "src/cnn/tensor.cpp"
)

target_include_directories(test PRIVATE
${PROJECT_SOURCE_DIR}/includes/cnn
${CMAKE_BINARY_DIR}/generated
)