cmake_minimum_required(VERSION 3.12)
project(SofaTopologyMapping LANGUAGES CXX)

set(SOFATOPOLOGYMAPPING_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/CenterPointTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Edge2QuadTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2QuadTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2TetraTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/IdentityTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.inl
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Quad2TriangleTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedHexaTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.inl
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/SubsetTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Tetra2TriangleTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/Triangle2EdgeTopologicalMapping.h
    ${SOFATOPOLOGYMAPPING_SRC}/config.h.in
    ${SOFATOPOLOGYMAPPING_SRC}/initSofaTopologyMapping.h
)

set(SOURCE_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/CenterPointTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Edge2QuadTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2QuadTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Hexa2TetraTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/IdentityTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointMechanicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Mesh2PointTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Quad2TriangleTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedHexaTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraMechanicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/SimpleTesselatedTetraTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/SubsetTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Tetra2TriangleTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/Triangle2EdgeTopologicalMapping.cpp
    ${SOFATOPOLOGYMAPPING_SRC}/initSofaTopologyMapping.cpp
)

sofa_find_package(SofaGeneralSimpleFem REQUIRED)
sofa_find_package(SofaGeneralTopology REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGeneralTopology SofaGeneralSimpleFem)

sofa_configure_unity_build(TARGET ${PROJECT_NAME}
                           ACTIVATED "${SOFA_BUILD_WITH_UNITY_BUILD_ENABLED}"
                           BATCHSIZE "${SOFA_UNITY_BUILD_BATCH_SIZE}")

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
