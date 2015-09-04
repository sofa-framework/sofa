cmake_minimum_required(VERSION 2.8)
project(gtest CXX C)

## Options
# When other libraries are using a shared version of runtime libraries,
# Google Test also has to use one.
option(gtest_force_shared_crt "Use shared (DLL) run-time lib even when Google Test is built as static lib." ON)
mark_as_advanced(gtest_force_shared_crt)

option(gtest_disable_pthreads "Disable uses of pthreads in gtest." OFF)
mark_as_advanced(gtest_disable_pthreads)

## Config
# Define helper functions and macros used by Google Test.
include(cmake/internal_utils.cmake)
config_compiler_and_linker()  # Defined in internal_utils.cmake.

include_directories(include)

# The code must be relocatable if we want to link a shared library against it.
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    add_compile_options(-fPIC)
endif()

cxx_library(gtest "${cxx_strict}" src/gtest-all.cc)
cxx_library(gtest_main "${cxx_strict}" src/gtest_main.cc)
target_link_libraries(gtest_main gtest)

# In Visual Studio 2012 (VC11) _VARIADIC_MAX is by default defined as
# 5, which is not enough for gtest.  So we ensure that everything that
# links against gtest defines _VARIADIC_MAX=10
if(MSVC)
    target_compile_options(gtest PUBLIC "-D_VARIADIC_MAX=10")
    target_compile_options(gtest_main PUBLIC "-D_VARIADIC_MAX=10")
endif()

target_include_directories(gtest PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(gtest_main PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")


## Installation rules
install(DIRECTORY include/ DESTINATION include)
install(TARGETS gtest gtest_main
        EXPORT GTestTargets
        COMPONENT gtest_libraries
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)


### GTestConfig.cmake

include(CMakePackageConfigHelpers)

## GTestConfig.cmake (build tree)
set(GTEST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
configure_package_config_file(GTestConfig.cmake.in
                              GTestConfig.cmake
                              INSTALL_DESTINATION none
                              PATH_VARS GTEST_INCLUDE_DIR)

## GTestConfig.cmake (installed)
set(GTEST_INCLUDE_DIR include)
configure_package_config_file(GTestConfig.cmake.in
                              GTestConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/GTest
                              PATH_VARS GTEST_INCLUDE_DIR)
install(EXPORT GTestTargets DESTINATION lib/cmake/GTest) # GTestTargets.cmake
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GTestConfigInstall.cmake"
        DESTINATION lib/cmake/GTest
        RENAME GTestConfig.cmake)
