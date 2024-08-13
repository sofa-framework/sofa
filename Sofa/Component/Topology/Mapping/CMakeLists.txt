cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.Topology.Mapping LANGUAGES CXX)

set(SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR "src/sofa/component/topology/mapping")

set(HEADER_FILES
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/init.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/CenterPointTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Edge2QuadTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Hexa2QuadTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Hexa2TetraTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/IdentityTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Quad2TriangleTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/SubsetTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Tetra2TriangleTopologicalMapping.h
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Triangle2EdgeTopologicalMapping.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/CenterPointTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Edge2QuadTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Hexa2QuadTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Hexa2TetraTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/IdentityTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Quad2TriangleTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/SubsetTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Tetra2TriangleTopologicalMapping.cpp
    ${SOFACOMPONENTTOPOLOGYMAPPING_SOURCE_DIR}/Triangle2EdgeTopologicalMapping.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core Sofa.Component.Topology.Container.Grid Sofa.Component.Topology.Container.Dynamic)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
