cmake_minimum_required(VERSION 3.20)

#set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_61")

project(small_la LANGUAGES CXX)

add_subdirectory(src)
add_subdirectory(app)