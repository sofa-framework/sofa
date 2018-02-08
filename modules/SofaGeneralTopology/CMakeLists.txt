cmake_minimum_required(VERSION 3.1)
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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_TOPOLOGY")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
