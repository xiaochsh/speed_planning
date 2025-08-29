cmake_minimum_required(VERSION 3.18)
project(speed_optimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(OsqpEigen REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_executable(speed_optimizer test_speed_optimizer.cc)
target_link_libraries(speed_optimizer OsqpEigen::OsqpEigen Matplot++::matplot Python3::Python)
