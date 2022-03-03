cmake_minimum_required(VERSION 3.16)
project(classifier)

add_definitions("-std=c++11")
# add_definitions("-g")  # for debugging
set(CMAKE_CXX_STANDARD 20)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# set(ALLOW_TRANSPARENT_CIPHERTEXT true)
# set(SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT OFF)
find_package(SEAL 3.7.1 REQUIRED)

add_executable(classifier main.cpp Layer.cpp Network.cpp Network.h)
target_link_libraries(classifier ${CONAN_LIBS} SEAL::seal)
