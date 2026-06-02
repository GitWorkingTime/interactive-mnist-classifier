cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)
# ── Main Executable ───────────────────────────────────────────────────────────

add_executable(main
# ── Add Files here ────────────────────────────────────────────────────────────
    "src/main.cpp"
)

target_include_directories(main PRIVATE
    includes
    ${CMAKE_BINARY_DIR}/generated
)