cmake_minimum_required(VERSION 2.8.12)
project(SofaLoader)

set(HEADER_FILES
    MeshObjLoader.h
    MeshVTKLoader.h
    config.h
    initLoader.h
)

set(SOURCE_FILES
    MeshObjLoader.cpp
    MeshVTKLoader.cpp
    initLoader.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree SofaHelper)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_LOADER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
