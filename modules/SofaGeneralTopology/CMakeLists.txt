cmake_minimum_required(VERSION 3.12)
project(SofaGeneralTopology)

set(HEADER_FILES
    config.h
    initGeneralTopology.h
    )

set(SOURCE_FILES
    initGeneralTopology.cpp
    )

list(APPEND HEADER_FILES
    CubeTopology.h
    CylinderGridTopology.h
    SphereGridTopology.h
    SphereQuadTopology.h
    )
list(APPEND SOURCE_FILES
    CubeTopology.cpp
    CylinderGridTopology.cpp
    SphereGridTopology.cpp
    SphereQuadTopology.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaBaseTopology)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGeneral
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
