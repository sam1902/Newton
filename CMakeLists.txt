cmake_minimum_required(VERSION 3.15)
project(NewtonTests)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/include/eigen3/)

add_executable(NewtonTests main.cpp Newton.cpp Newton.h Tests.cpp Tests.h)

project(Newton)
include_directories(/usr/local/include/eigen3/)
add_library(Newton Newton.cpp Newton.h)