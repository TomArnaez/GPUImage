cmake_minimum_required(VERSION 3.28.2)

project(GPUImage CXX CUDA)

set(KOKKOS_DIR "${CMAKE_SOURCE_DIR}/Kokkos/lib/cmake/Kokkos")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)

message(STATUS "Kokkos directory: ${KOKKOS_DIR}")

add_executable(GPUImage)

find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})
find_package(OpenCV CONFIG REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS ")matplotlibcpp.h")
find_package(Python3 COMPONENTS Development REQUIRED)

target_sources(GPUImage
  PRIVATE
    main.cpp
  PUBLIC
    include/concepts.hpp
    include/image.hpp
    include/transforms.hpp
    include/statistics.hpp
)
set_target_properties(GPUImage PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
get_target_property(source_files GPUImage SOURCES)
set_source_files_properties(${source_files} PROPERTIES LANGUAGE CUDA)
target_link_libraries(GPUImage Kokkos::kokkos ${OpenCV_LIBS} ${Python3_LIBRARIES})
target_include_directories(GPUImage PUBLIC ${Kokkos_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${MATPLOTLIB_CPP_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})