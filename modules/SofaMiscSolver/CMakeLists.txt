cmake_minimum_required(VERSION 3.1)
project(SofaMiscSolver VERSION 1.0 LANGUAGES CXX)

find_package(SofaSimulation REQUIRED)

set(SOFAMISCSOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCSOLVER_SRC}/config.h.in
    ${SOFAMISCSOLVER_SRC}/initMiscSolver.h
    ${SOFAMISCSOLVER_SRC}/DampVelocitySolver.h
    ${SOFAMISCSOLVER_SRC}/NewmarkImplicitSolver.h
)

set(SOURCE_FILES
    ${SOFAMISCSOLVER_SRC}/initMiscSolver.cpp
    ${SOFAMISCSOLVER_SRC}/DampVelocitySolver.cpp
    ${SOFAMISCSOLVER_SRC}/NewmarkImplicitSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src/${PROJECT_NAME}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )
# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCSOLVER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCSOLVER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscSolver_test)
endif()

