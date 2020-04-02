cmake_minimum_required(VERSION 3.1)
project(SofaMiscExtra)

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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_EXTRA")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} "SofaMiscExtra/${PROJECT_NAME}")
