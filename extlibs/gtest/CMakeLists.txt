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

set(HEADER_FILES
    include/gtest/gtest.h
    include/gtest/gtest-death-test.h
    include/gtest/gtest-message.h
    include/gtest/gtest-param-test.h
    include/gtest/gtest_pred_impl.h
    include/gtest/gtest-printers.h
    include/gtest/gtest_prod.h
    include/gtest/gtest-spi.h
    include/gtest/gtest-test-part.h
    include/gtest/gtest-typed-test.h
)

set(SOURCE_FILES
    src/gtest.cc
    src/gtest-death-test.cc
    src/gtest-filepath.cc
    src/gtest-port.cc
    src/gtest-printers.cc
    src/gtest-test-part.cc
    src/gtest-typed-test.cc
    )

# make the gtest library as a shared library.
cxx_shared_library(gtest "${cxx_strict}" ${SOURCE_FILES} ${HEADER_FILES})

target_compile_options(gtest INTERFACE "-DGTEST_LINKED_AS_SHARED_LIBRARY=1")
target_compile_options(gtest PRIVATE "-DGTEST_CREATE_SHARED_LIBRARY=1")

# In Visual Studio 2012 (VC11) _VARIADIC_MAX is by default defined as
# 5, which is not enough for gtest.  So we ensure that everything that
# links against gtest defines _VARIADIC_MAX=10
if(MSVC)
    target_compile_options(gtest PUBLIC "-D_VARIADIC_MAX=10")
else()
    target_compile_options(gtest PRIVATE "-Wno-missing-field-initializers")
endif()

target_include_directories(gtest PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(gtest PUBLIC "$<INSTALL_INTERFACE:include>")

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_create_package(GTest 2.6.2 "gtest" "")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gtest
        COMPONENT GTest_headers
        DESTINATION include/)
