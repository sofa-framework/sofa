cmake_minimum_required(VERSION 3.12)
project(Sofa.Testing LANGUAGES CXX)

# Enable testing features of cmake, like the add_test() command.
enable_testing()

# add gtest library
# set(gtest_disable_pthreads ON CACHE INTERNAL "")
add_subdirectory(extlibs/gtest)
# add custom gtest main
add_subdirectory(SofaGTestMain)

set(SOFATESTINGSRC_ROOT "src/sofa/testing")

set(HEADER_FILES
    ${SOFATESTINGSRC_ROOT}/config.h.in
    ${SOFATESTINGSRC_ROOT}/initSofa.Testing.h
    ${SOFATESTINGSRC_ROOT}/BaseTest.h
    ${SOFATESTINGSRC_ROOT}/LinearCongruentialRandomGenerator.h
    ${SOFATESTINGSRC_ROOT}/NumericTest.h
    ${SOFATESTINGSRC_ROOT}/TestMessageHandler.h
    ${SOFATESTINGSRC_ROOT}/BaseSimulationTest.h
)

set(SOURCE_FILES
    ${SOFATESTINGSRC_ROOT}/initSofa.Testing.cpp
    ${SOFATESTINGSRC_ROOT}/BaseTest.cpp
    ${SOFATESTINGSRC_ROOT}/LinearCongruentialRandomGenerator.cpp
    ${SOFATESTINGSRC_ROOT}/NumericTest.cpp
    ${SOFATESTINGSRC_ROOT}/TestMessageHandler.cpp
    ${SOFATESTINGSRC_ROOT}/BaseSimulationTest.cpp
)


sofa_find_package(Sofa.Helper REQUIRED)
sofa_find_package(Sofa.DefaultType REQUIRED)
sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Simulation.Graph REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper Sofa.DefaultType Sofa.Core Sofa.Simulation.Graph gtest SofaGTestMain )
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
