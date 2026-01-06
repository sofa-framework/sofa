cmake_minimum_required(VERSION 3.22)
project(Sofa.Topology LANGUAGES CXX)

set(SOFATOPOLOGYSRC_ROOT "src/sofa/topology")

set(HEADER_FILES
    ${SOFATOPOLOGYSRC_ROOT}/config.h.in
    ${SOFATOPOLOGYSRC_ROOT}/init.h
    ${SOFATOPOLOGYSRC_ROOT}/Element.h
    ${SOFATOPOLOGYSRC_ROOT}/Topology.h
    ${SOFATOPOLOGYSRC_ROOT}/Point.h
    ${SOFATOPOLOGYSRC_ROOT}/Edge.h
    ${SOFATOPOLOGYSRC_ROOT}/Triangle.h
    ${SOFATOPOLOGYSRC_ROOT}/Quad.h
    ${SOFATOPOLOGYSRC_ROOT}/Tetrahedron.h
    ${SOFATOPOLOGYSRC_ROOT}/Pentahedron.h
    ${SOFATOPOLOGYSRC_ROOT}/Prism.h
    ${SOFATOPOLOGYSRC_ROOT}/Pyramid.h
    ${SOFATOPOLOGYSRC_ROOT}/Hexahedron.h
)

set(SOURCE_FILES
    ${SOFATOPOLOGYSRC_ROOT}/init.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Topology.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Element.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Point.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Edge.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Triangle.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Quad.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Tetrahedron.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Prism.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Pyramid.cpp
    ${SOFATOPOLOGYSRC_ROOT}/Hexahedron.cpp
)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Type REQUIRED)
sofa_find_package(Sofa.Geometry REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Type Sofa.Geometry)

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
cmake_dependent_option(SOFA_TOPOLOGY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_TOPOLOGY_BUILD_TESTS)
    add_subdirectory(test)
endif()
