cmake_minimum_required(VERSION 3.16)
project(classifier)

add_definitions("-g")  # gcc adds debug information (confer `man gcc`)
# add_definitions("-Wall")
set(CMAKE_CXX_STANDARD 23)
# include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
# conan_basic_setup()

find_package(SEAL 4.1.1 REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")
find_package(httplib REQUIRED)
find_package(plog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(xtensor REQUIRED)
find_package(lodepng REQUIRED)

add_executable(classifier main.cpp DenseLayer.cpp ActivationLayer.cpp Network.cpp)
add_executable(evaluate evaluate.cpp DenseLayer.cpp ActivationLayer.cpp Network.cpp)
target_link_libraries(classifier ${CONAN_LIBS} SEAL::seal ntl httplib::httplib plog::plog nlohmann_json::nlohmann_json xtensor lodepng::lodepng)
target_link_libraries(evaluate ${CONAN_LIBS} SEAL::seal ntl httplib::httplib plog::plog nlohmann_json::nlohmann_json xtensor lodepng::lodepng)
