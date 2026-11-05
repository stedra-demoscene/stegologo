cmake_minimum_required(VERSION 3.18)
project(stegojni LANGUAGES CXX)
# shared native decoder (core/): static lib stego_core + JNI shim
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../core core_build)
add_library(stegojni SHARED stego_jni.cpp)
target_link_libraries(stegojni PRIVATE stego_core log)
target_compile_options(stegojni PRIVATE -O3 -Wall)
