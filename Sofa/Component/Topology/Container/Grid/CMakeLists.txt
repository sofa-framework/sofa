cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.Topology.Container.Grid LANGUAGES CXX)

set(SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR "src/sofa/component/topology/container/grid")

set(HEADER_FILES
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/init.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/polygon_cube_intersection/polygon_cube_intersection.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/polygon_cube_intersection/vec.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/GridTopology.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/RegularGridTopology.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SparseGridTopology.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/CylinderGridTopology.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SphereGridTopology.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SparseGridMultipleTopology.h
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SparseGridRamificationTopology.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/polygon_cube_intersection/polygon_cube_intersection.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/GridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/RegularGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SparseGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/CylinderGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SphereGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SparseGridMultipleTopology.cpp
    ${SOFACOMPONENTTOPOLOGYCONTAINERGRID_SOURCE_DIR}/SparseGridRamificationTopology.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Constant REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core Sofa.Component.Topology.Container.Constant)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_TOPOLOGY_CONTAINER_GRID_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_TOPOLOGY_CONTAINER_GRID_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
