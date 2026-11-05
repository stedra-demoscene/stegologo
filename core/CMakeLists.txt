cmake_minimum_required(VERSION 3.18)
project(stego_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(stego_core STATIC
  src/asset.cpp src/image.cpp src/fit.cpp src/decode.cpp src/register.cpp src/api.cpp src/mark_detect.cpp src/mark_decode.cpp src/demo_decode.cpp)
target_include_directories(stego_core PUBLIC include)
# no -ffast-math: results are deterministic across compilers; loops are written to vectorise anyway
target_compile_options(stego_core PRIVATE -O3 -Wall -Wno-unused-function)

add_executable(stego_core_cli cli/stego_core_cli.cpp)
target_link_libraries(stego_core_cli PRIVATE stego_core)

option(STEGO_PYTHON "Build the pybind11 module geostego_core" OFF)
if(STEGO_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(geostego_core python/binding.cpp)
  target_include_directories(geostego_core PRIVATE src)
  target_link_libraries(geostego_core PRIVATE stego_core)
endif()
