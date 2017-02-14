cmake_minimum_required(VERSION 3.1)
project(SofaLoader)

set(HEADER_FILES
    BaseVTKReader.h
    BaseVTKReader.inl
    MeshObjLoader.h
    MeshVTKLoader.h
    config.h
    initLoader.h
)

set(SOURCE_FILES
    BaseVTKReader.cpp
    BaseVTKReader.inl
    MeshObjLoader.cpp
    MeshVTKLoader.cpp
    initLoader.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree SofaHelper)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_LOADER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
