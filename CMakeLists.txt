cmake_minimum_required(VERSION 3.17)
project(cuda_starter CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CUDA_ARCHITECTURES 75)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12/bin/nvcc)

# Add subdirectories
add_subdirectory(src)

# TODO:
# add_subdirectory(examples)
# add_subdirectory(tests)
# add_subdirectory(benchmark)
