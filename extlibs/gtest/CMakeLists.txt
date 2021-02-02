cmake_minimum_required(VERSION 3.12)
project(gtest VERSION 2.6.2 LANGUAGES CXX C)

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

# The code must be relocatable if we want to link a shared library against it.
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    add_compile_options(-fPIC)
endif()

set(HEADER_FILES
    include/gtest/gtest.h
    include/gtest/gtest-death-test.h
    include/gtest/gtest-matchers.h
    include/gtest/gtest-message.h
    include/gtest/gtest-param-test.h
    include/gtest/gtest_pred_impl.h
    include/gtest/gtest-printers.h
    include/gtest/gtest_prod.h
    include/gtest/gtest-spi.h
    include/gtest/gtest-test-part.h
    include/gtest/gtest-typed-test.h
    include/gtest/internal/gtest-death-test-internal.h
    include/gtest/internal/gtest-filepath.h
    include/gtest/internal/gtest-internal.h
    include/gtest/internal/gtest-linked_ptr.h
    include/gtest/internal/gtest-param-util-generated.h
    include/gtest/internal/gtest-param-util.h
    include/gtest/internal/gtest-port-arch.h
    include/gtest/internal/gtest-port.h
    include/gtest/internal/gtest-string.h
    include/gtest/internal/gtest-tuple.h
    include/gtest/internal/gtest-type-util.h
    include/gtest/internal/custom/gtest-port.h
    include/gtest/internal/custom/gtest-printers.h
    include/gtest/internal/custom/gtest.h
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

target_include_directories(gtest
    SYSTEM PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>" # for src/gtest.cc -> src/gtest-internal-inl.h
    )

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_create_package_with_targets(
    PACKAGE_NAME GTest
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "include"
    INCLUDE_INSTALL_DIR "extlibs/GTest"
    )
