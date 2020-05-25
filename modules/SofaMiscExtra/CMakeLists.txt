cmake_minimum_required(VERSION 3.1)
project(SofaMiscExtra)

find_package(SofaGeneral REQUIRED) # SofaGeneralMeshCollision SofaTopologyMapping

set(HEADER_FILES
    initMiscExtra.h
    )

set(SOURCE_FILES
    initMiscExtra.cpp
    )

list(APPEND HEADER_FILES
    MeshTetraStuffing.h
    )
list(APPEND SOURCE_FILES
    MeshTetraStuffing.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGeneralMeshCollision SofaMiscTopology SofaTopologyMapping)

sofa_install_targets(SofaMisc ${PROJECT_NAME} "SofaMiscExtra/${PROJECT_NAME}")
