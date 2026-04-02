cmake_minimum_required(VERSION 3.20)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# ── CUDA architecture targets ──────────────────────────────────────────────
# Ampere (A10G used by AWS g5), Turing (T4 used by AWS g4dn)
set(CMAKE_CUDA_ARCHITECTURES "75;86")

# ── Build shared library consumed by Python ctypes / cffi ─────────────────
add_library(cuda_kernels SHARED
    matrix_ops.cu
)

target_include_directories(cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
    OUTPUT_NAME "cuda_kernels"
)

target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --generate-line-info
    >
)

install(TARGETS cuda_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES matrix_ops.h DESTINATION include)
