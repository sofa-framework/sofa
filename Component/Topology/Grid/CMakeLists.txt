cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Topology.Grid LANGUAGES CXX)

set(SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR "src/sofa/component/topology/grid")

set(HEADER_FILES
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/init.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/polygon_cube_intersection/polygon_cube_intersection.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/polygon_cube_intersection/vec.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/GridTopology.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/RegularGridTopology.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SparseGridTopology.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/CylinderGridTopology.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SphereGridTopology.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SparseGridMultipleTopology.h
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SparseGridRamificationTopology.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/polygon_cube_intersection/polygon_cube_intersection.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/GridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/RegularGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SparseGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/CylinderGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SphereGridTopology.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SparseGridMultipleTopology.cpp
    ${SOFACOMPONENTTOPOLOGYGRID_SOURCE_DIR}/SparseGridRamificationTopology.cpp
)

sofa_find_package(SofaSimulationCore REQUIRED)
sofa_find_package(Sofa.Component.Topology.StaticContainer REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore Sofa.Component.Topology.StaticContainer)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
