cmake_minimum_required(VERSION 3.16)
project(classifier)

add_definitions("-g")  # gcc adds debug information (confer `man gcc`)
# add_definitions("-Wall")
set(CMAKE_CXX_STANDARD 23)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(SEAL 4.0.0 REQUIRED)

add_executable(classifier main.cpp DenseLayer.cpp ActivationLayer.cpp Network.cpp)
add_executable(evaluate evaluate.cpp DenseLayer.cpp ActivationLayer.cpp Network.cpp)
target_link_libraries(classifier ${CONAN_LIBS} SEAL::seal ntl)
target_link_libraries(evaluate ${CONAN_LIBS} SEAL::seal ntl)
