cmake_minimum_required(VERSION 3.22)
project(Sofa.FEM LANGUAGES CXX)

set(SOFAFEMSRC_ROOT "src/sofa/fem")

set(HEADER_FILES
    ${SOFAFEMSRC_ROOT}/config.h.in
    ${SOFAFEMSRC_ROOT}/init.h

    ${SOFAFEMSRC_ROOT}/FiniteElement.h
    ${SOFAFEMSRC_ROOT}/FiniteElement[all].h
    ${SOFAFEMSRC_ROOT}/FiniteElement[Edge].h
    ${SOFAFEMSRC_ROOT}/FiniteElement[Hexahedron].h
    ${SOFAFEMSRC_ROOT}/FiniteElement[Quad].h
    ${SOFAFEMSRC_ROOT}/FiniteElement[Tetrahedron].h
    ${SOFAFEMSRC_ROOT}/FiniteElement[Triangle].h
)

set(SOURCE_FILES
    ${SOFAFEMSRC_ROOT}/init.cpp

    ${SOFAFEMSRC_ROOT}/FiniteElement[Edge].cpp
    ${SOFAFEMSRC_ROOT}/FiniteElement[Hexahedron].cpp
    ${SOFAFEMSRC_ROOT}/FiniteElement[Quad].cpp
    ${SOFAFEMSRC_ROOT}/FiniteElement[Tetrahedron].cpp
    ${SOFAFEMSRC_ROOT}/FiniteElement[Triangle].cpp
)

find_package(Sofa.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core)

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
cmake_dependent_option(SOFA_FEM_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_FEM_BUILD_TESTS)
    add_subdirectory(test)
endif()
