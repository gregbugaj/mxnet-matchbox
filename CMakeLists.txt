cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(matchbox C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src)

#SET(ENV{MXNET_ROOT} "/home/gbugaj/dev/3rdparty/mxnet")
SET(ENV{MXNET_ROOT} "/home/gbugaj/dev/3rdparty/mxnet")

add_subdirectory(src)
