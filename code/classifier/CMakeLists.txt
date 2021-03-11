cmake_minimum_required(VERSION 3.16)
project(classifier)

add_definitions("-std=c++11")
set(CMAKE_CXX_STANDARD 20)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(classifier main.cpp Layer.cpp Network.cpp Network.h)
target_link_libraries(classifier ${CONAN_LIBS})
