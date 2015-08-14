########################################################################
# CMake build script for Google Test.

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fPIC)
endif()

# BUILD_SHARED_LIBS is a standard CMake variable, but we declare it here to
# make it prominent in the GUI.
option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." OFF)

# When other libraries are using a shared version of runtime libraries,
# Google Test also has to use one.
option(gtest_force_shared_crt "Use shared (DLL) run-time lib even when Google Test is built as static lib." OFF)
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
                              ${CMAKE_BINARY_DIR}/GTest/GTestConfig.cmake
                              INSTALL_DESTINATION GTest
                              PATH_VARS GTEST_INCLUDE_DIR)

# GTestConfig.cmake (install directory)
set(GTEST_INCLUDE_DIR include)
configure_package_config_file(GTestConfig.cmake.in
                              ${CMAKE_BINARY_DIR}/GTest/GTestConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/GTest
                              PATH_VARS GTEST_INCLUDE_DIR)

install(FILES ${CMAKE_BINARY_DIR}/GTest/GTestConfigInstall.cmake
        DESTINATION lib/cmake/GTest
        RENAME GTestConfig.cmake)
