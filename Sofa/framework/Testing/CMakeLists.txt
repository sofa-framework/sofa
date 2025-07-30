cmake_minimum_required(VERSION 3.22)
project(Sofa.Testing LANGUAGES CXX)

# Enable testing features of cmake, like the add_test() command.
enable_testing()

# add googletest library
# set(googletest_disable_pthreads ON CACHE INTERNAL "")
find_package(GTest QUIET)
if(NOT GTest_FOUND AND SOFA_ALLOW_FETCH_DEPENDENCIES)
    message("${PROJECT_NAME}: DEPENDENCY googletest NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is ON, fetching googletest...")

    set(BUILD_GMOCK OFF CACHE INTERNAL "")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE) # rely on SOFA macros
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(gtest_disable_pthreads OFF CACHE BOOL "" FORCE)

    message("${PROJECT_NAME}: adding subdirectory ${googletest_SOURCE_DIR}")

    sofa_fetch_dependency(googletest
            GIT_REPOSITORY https://github.com/google/googletest
            GIT_TAG        v1.14.0
    )

    target_compile_options(gtest_main PRIVATE "-DGTEST_LINKED_AS_SHARED_LIBRARY=0")
    target_compile_options(gtest PRIVATE "-DGTEST_CREATE_SHARED_LIBRARY=1")

    install(DIRECTORY ${googletest_SOURCE_DIR}/googletest/include/gtest DESTINATION include/extlibs/GTest/ COMPONENT headers)
    target_include_directories(gtest PUBLIC "$<INSTALL_INTERFACE:include/extlibs/GTest/>")

    include(SofaMacros)
    sofa_create_package_with_targets(
        PACKAGE_NAME GTest
        PACKAGE_VERSION 1.14.0
        TARGETS gtest AUTO_SET_TARGET_PROPERTIES
        INCLUDE_SOURCE_DIR "include"
        INCLUDE_INSTALL_DIR "extlibs/GTest"
        )

    set_target_properties(gtest PROPERTIES FOLDER Testing)
    set_target_properties(gtest_main PROPERTIES FOLDER Testing)

elseif (NOT GTest_FOUND)
    message(FATAL_ERROR "${PROJECT_NAME}: DEPENDENCY googletest NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is OFF and thus cannot be fetched. Install googletest, or enable SOFA_ALLOW_FETCH_DEPENDENCIES to fix this issue.")
endif()



# add custom googletest main
add_subdirectory(SofaGTestMain)

set(SOFATESTINGSRC_ROOT "src/sofa/testing")

set(HEADER_FILES
    ${SOFATESTINGSRC_ROOT}/config.h.in
    ${SOFATESTINGSRC_ROOT}/initSofa.Testing.h
    ${SOFATESTINGSRC_ROOT}/BaseTest.h
    ${SOFATESTINGSRC_ROOT}/LinearCongruentialRandomGenerator.h
    ${SOFATESTINGSRC_ROOT}/NumericTest.h
    ${SOFATESTINGSRC_ROOT}/ScopedPlugin.h
    ${SOFATESTINGSRC_ROOT}/TestMessageHandler.h
    ${SOFATESTINGSRC_ROOT}/BaseSimulationTest.h
)

set(SOURCE_FILES
    ${SOFATESTINGSRC_ROOT}/initSofa.Testing.cpp
    ${SOFATESTINGSRC_ROOT}/BaseTest.cpp
    ${SOFATESTINGSRC_ROOT}/LinearCongruentialRandomGenerator.cpp
    ${SOFATESTINGSRC_ROOT}/NumericTest.cpp
    ${SOFATESTINGSRC_ROOT}/ScopedPlugin.cpp
    ${SOFATESTINGSRC_ROOT}/TestMessageHandler.cpp
    ${SOFATESTINGSRC_ROOT}/BaseSimulationTest.cpp
)


sofa_find_package(Sofa.Helper REQUIRED)
sofa_find_package(Sofa.DefaultType REQUIRED)
sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(Sofa.SimpleApi REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper Sofa.DefaultType Sofa.Core Sofa.Simulation.Graph Sofa.SimpleApi GTest::gtest SofaGTestMain )
target_compile_options(${PROJECT_NAME} PUBLIC "-DGTEST_LINKED_AS_SHARED_LIBRARY=1")

set(SOFA_TESTING_RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_TESTING_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_TESTING_BUILD_TESTS)
    add_subdirectory(test)
endif()
