# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.8)
   message(FATAL_ERROR "CMake >= 2.8.0 required")
endif()
if(CMAKE_VERSION VERSION_LESS "2.8.3")
   message(FATAL_ERROR "CMake >= 2.8.3 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.8.3...3.26)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Protect against multiple inclusion, which would fail when already imported targets are added once more.
set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS Kokkos::CUDA Kokkos::kokkoscore Kokkos::kokkoscontainers Kokkos::kokkosalgorithms Kokkos::kokkossimd Kokkos::kokkos)
  list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
  if(TARGET "${_cmake_expected_target}")
    list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
  else()
    list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
  endif()
endforeach()
unset(_cmake_expected_target)
if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
  unset(_cmake_targets_defined)
  unset(_cmake_targets_not_defined)
  unset(_cmake_expected_targets)
  unset(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()
if(NOT _cmake_targets_defined STREQUAL "")
  string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
  string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)


# Compute the installation prefix relative to this file.
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

# Create imported target Kokkos::CUDA
add_library(Kokkos::CUDA INTERFACE IMPORTED)

set_target_properties(Kokkos::CUDA PROPERTIES
  INTERFACE_LINK_LIBRARIES "CUDA::cuda_driver;CUDA::cudart"
)

# Create imported target Kokkos::kokkoscore
add_library(Kokkos::kokkoscore STATIC IMPORTED)

set_target_properties(Kokkos::kokkoscore PROPERTIES
  INTERFACE_COMPILE_DEFINITIONS "\$<\$<COMPILE_LANGUAGE:CUDA>:KOKKOS_DEPENDENCE>"
  INTERFACE_COMPILE_FEATURES "cxx_std_17"
  INTERFACE_COMPILE_OPTIONS "\$<\$<COMPILE_LANGUAGE:CUDA>:>;\$<\$<COMPILE_LANGUAGE:CUDA>:-extended-lambda;-Wext-lambda-captures-this>;\$<\$<COMPILE_LANGUAGE:CUDA>:>"
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "Kokkos::CUDA"
)

# Create imported target Kokkos::kokkoscontainers
add_library(Kokkos::kokkoscontainers STATIC IMPORTED)

set_target_properties(Kokkos::kokkoscontainers PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "Kokkos::kokkoscore"
)

# Create imported target Kokkos::kokkosalgorithms
add_library(Kokkos::kokkosalgorithms INTERFACE IMPORTED)

set_target_properties(Kokkos::kokkosalgorithms PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
)

# Create imported target Kokkos::kokkossimd
add_library(Kokkos::kokkossimd STATIC IMPORTED)

set_target_properties(Kokkos::kokkossimd PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
)

# Create imported target Kokkos::kokkos
add_library(Kokkos::kokkos INTERFACE IMPORTED)

set_target_properties(Kokkos::kokkos PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
  INTERFACE_LINK_LIBRARIES "Kokkos::kokkoscore;Kokkos::kokkoscontainers;Kokkos::kokkosalgorithms;Kokkos::kokkossimd"
)

if(CMAKE_VERSION VERSION_LESS 3.0.0)
  message(FATAL_ERROR "This file relies on consumers using CMake 3.0.0 or greater.")
endif()

# Load information for each installed configuration.
file(GLOB _cmake_config_files "${CMAKE_CURRENT_LIST_DIR}/KokkosTargets-*.cmake")
foreach(_cmake_config_file IN LISTS _cmake_config_files)
  include("${_cmake_config_file}")
endforeach()
unset(_cmake_config_file)
unset(_cmake_config_files)

# Cleanup temporary variables.
set(_IMPORT_PREFIX)

# Loop over all imported files and verify that they actually exist
foreach(_cmake_target IN LISTS _cmake_import_check_targets)
  foreach(_cmake_file IN LISTS "_cmake_import_check_files_for_${_cmake_target}")
    if(NOT EXISTS "${_cmake_file}")
      message(FATAL_ERROR "The imported target \"${_cmake_target}\" references the file
   \"${_cmake_file}\"
but this file does not exist.  Possible reasons include:
* The file was deleted, renamed, or moved to another location.
* An install or uninstall procedure did not complete successfully.
* The installation package was faulty and contained
   \"${CMAKE_CURRENT_LIST_FILE}\"
but not all the files it references.
")
    endif()
  endforeach()
  unset(_cmake_file)
  unset("_cmake_import_check_files_for_${_cmake_target}")
endforeach()
unset(_cmake_target)
unset(_cmake_import_check_targets)

# This file does not depend on other imported targets which have
# been exported from the same project but in a separate export set.

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
