# Inkling 2 Mini engine (strata-inkling2) for one NVIDIA card plus system RAM.
#   tools\build.bat             release build (CUDA 7.5 / 8.6 / 8.9 / 12.0, AVX2), into build\bin
#   tools\build.bat --native    this PC's GPU and CPU only (faster to build)
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Turing, Ampere (RTX 30), Ada (RTX 40), Blackwell (RTX 50), and PTX for newer cards
  set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual" CACHE STRING "CUDA architectures")
endif()
project(inkling2_mini LANGUAGES C CXX CUDA)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(INK_GGML_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp CACHE PATH "llama.cpp tree to take ggml from")
add_subdirectory(engine)
