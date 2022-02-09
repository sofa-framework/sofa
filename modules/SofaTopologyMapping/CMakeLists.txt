cmake_minimum_required(VERSION 3.12)
project(SofaTopologyMapping LANGUAGES CXX)

set(SOFATOPOLOGYMAPPING_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/config.h.in
    ${SOFATOPOLOGYMAPPING_SRC}/initSofaTopologyMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.inl
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedHexaTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.inl
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointTopologicalMapping.h
)

set(SOURCE_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/initSofaTopologyMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedHexaTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraTopologicalMapping.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Topology.Mapping.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Topology.Mapping REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Mapping)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
