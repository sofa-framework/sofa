########################################################################
# CMake build script for Google Test.

# The code must be relocatable if we want to link a shared library against it.
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    add_compile_options(-fPIC)
endif()

if(WIN32)
    # MSVC2012 has some troubles with the way gtest use the STL, this preprocessor macro fix this issue
    add_definitions("-D_VARIADIC_MAX=10")
endif()

# BUILD_SHARED_LIBS is a standard CMake variable, but we declare it here to
# make it prominent in the GUI.
option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." OFF)

# When other libraries are using a shared version of runtime libraries,
# Google Test also has to use one.
option(gtest_force_shared_crt "Use shared (DLL) run-time lib even when Google Test is built as static lib." ON)
mark_as_advanced(gtest_force_shared_crt)

option(gtest_disable_pthreads "Disable uses of pthreads in gtest." OFF)
mark_as_advanced(gtest_disable_pthreads)

# Language "C" is required for find_package(Threads).
project(gtest CXX C)
cmake_minimum_required(VERSION 2.6.2)

# Define helper functions and macros used by Google Test.
include(cmake/internal_utils.cmake)

config_compiler_and_linker()  # Defined in internal_utils.cmake.

# Where Google Test's .h files can be found.
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}/../include ${gtest_SOURCE_DIR})

cxx_library(gtest "${cxx_strict}" src/gtest-all.cc)
cxx_library(gtest_main "${cxx_strict}" src/gtest_main.cc)
target_link_libraries(gtest_main gtest)



include(CMakePackageConfigHelpers)

target_include_directories(gtest PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(gtest_main PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

# The trailing slash is important
install(DIRECTORY include/ DESTINATION include)

install(TARGETS gtest gtest_main
        EXPORT GTestTargets
        COMPONENT gtest_libraries
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# GTestTargets.cmake
install(EXPORT GTestTargets
        DESTINATION lib/cmake/GTest)

# GTestConfig.cmake (build tree)
set(GTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
configure_package_config_file(GTestConfig.cmake.in
                              GTestConfig.cmake
                              INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
                              PATH_VARS GTEST_INCLUDE_DIR)

# GTestConfig.cmake (install directory)
set(GTEST_INCLUDE_DIR include)
configure_package_config_file(GTestConfig.cmake.in
                              GTestConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/GTest
                              PATH_VARS GTEST_INCLUDE_DIR)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GTestConfigInstall.cmake"
        DESTINATION lib/cmake/GTest
        RENAME GTestConfig.cmake)
