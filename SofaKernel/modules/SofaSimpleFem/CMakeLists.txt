cmake_minimum_required(VERSION 3.12)
project(SofaSimpleFem LANGUAGES CXX)

set(SOFASIMPLEFEM_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFASIMPLEFEM_SRC}/config.h.in
    ${SOFASIMPLEFEM_SRC}/initSofaSimpleFem.h
    ${SOFASIMPLEFEM_SRC}/HexahedronFEMForceField.h
    ${SOFASIMPLEFEM_SRC}/HexahedronFEMForceField.inl
    ${SOFASIMPLEFEM_SRC}/TetrahedronFEMForceField.h
    ${SOFASIMPLEFEM_SRC}/TetrahedronFEMForceField.inl
    ${SOFASIMPLEFEM_SRC}/TetrahedronDiffusionFEMForceField.h
    ${SOFASIMPLEFEM_SRC}/TetrahedronDiffusionFEMForceField.inl
)

set(SOURCE_FILES
    ${SOFASIMPLEFEM_SRC}/initSofaSimpleFem.cpp
    ${SOFASIMPLEFEM_SRC}/HexahedronFEMForceField.cpp
    ${SOFASIMPLEFEM_SRC}/TetrahedronFEMForceField.cpp
    ${SOFASIMPLEFEM_SRC}/TetrahedronDiffusionFEMForceField.cpp
)

sofa_find_package(SofaBase REQUIRED) # SofaBaseTopology
sofa_find_package(SofaBaseLinearSolver REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseLinearSolver)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFASIMPLEFEM_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFASIMPLEFEM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
