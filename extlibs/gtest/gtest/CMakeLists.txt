cmake_minimum_required(VERSION 3.1)
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

include_directories(include "${CMAKE_CURRENT_SOURCE_DIR}")

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
else()
    target_compile_options(gtest PRIVATE "-Wno-missing-field-initializers")
    target_compile_options(gtest_main PRIVATE "-Wno-missing-field-initializers")
endif()

target_include_directories(gtest PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(gtest_main PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(gtest PUBLIC "$<INSTALL_INTERFACE:include>")
target_include_directories(gtest_main PUBLIC "$<INSTALL_INTERFACE:include>")

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_create_package(GTest 2.6.2 "gtest;gtest_main" "")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gtest
        COMPONENT GTest_headers
        DESTINATION include/)
