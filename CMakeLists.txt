cmake_minimum_required(VERSION 3.29)

project(GPUImage)

set(KOKKOS_DIR "${CMAKE_SOURCE_DIR}/Kokkos/lib/cmake/Kokkos")

message(STATUS "Kokkos directory: ${KOKKOS_DIR}")

add_executable(GPUImage)

find_package(Kokkos REQUIRED PATHS ${KOKKOS_DIR})

target_sources(GPUImage
  PRIVATE
    main.cpp
    image.cpp
  PUBLIC
    include/concepts.hpp
    include/transforms.hpp
    include/statistics.hpp
)

target_compile_features(GPUImage PRIVATE cxx_std_20)
target_link_libraries(GPUImage Kokkos::kokkos)
target_include_directories(GPUImage PUBLIC ${Kokkos_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)