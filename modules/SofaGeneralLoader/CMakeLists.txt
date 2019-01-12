cmake_minimum_required(VERSION 3.1)
project(SofaGeneralLoader)

set(HEADER_FILES
    ReadState.h
    ReadState.inl
    ReadTopology.h
    ReadTopology.inl
    config.h
    initGeneralLoader.h
    )

set(SOURCE_FILES
    ReadState.cpp
    ReadTopology.cpp
    initGeneralLoader.cpp
    )

list(APPEND HEADER_FILES
    GIDMeshLoader.h
    GridMeshCreator.h
    InputEventReader.h
    MeshGmshLoader.h
    MeshOffLoader.h
    MeshSTLLoader.h
    MeshTrianLoader.h
    MeshXspLoader.h
    OffSequenceLoader.h
    SphereLoader.h
    StringMeshCreator.h
    VoxelGridLoader.h
    )
list(APPEND SOURCE_FILES
    GIDMeshLoader.cpp
    GridMeshCreator.cpp
    InputEventReader.cpp
    MeshGmshLoader.cpp
    MeshOffLoader.cpp
    MeshSTLLoader.cpp
    MeshTrianLoader.cpp
    MeshXspLoader.cpp
    OffSequenceLoader.cpp
    SphereLoader.cpp
    StringMeshCreator.cpp
    VoxelGridLoader.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationTree SofaHelper)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_LOADER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
