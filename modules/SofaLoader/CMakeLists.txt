cmake_minimum_required(VERSION 2.8.12)
project(SofaLoader)

set(HEADER_FILES
    GIDMeshLoader.h
    GridMeshCreator.h
    InputEventReader.h
    MeshGmshLoader.h
    MeshObjLoader.h
    MeshOffLoader.h
    MeshSTLLoader.h
    MeshTrianLoader.h
    MeshVTKLoader.h
    MeshXspLoader.h
    OffSequenceLoader.h
    ReadState.h
    ReadState.inl
    ReadTopology.h
    ReadTopology.inl
    SphereLoader.h
    StringMeshCreator.h
    VoxelGridLoader.h
    config.h
    initLoader.h
)

set(SOURCE_FILES
    GIDMeshLoader.cpp
    GridMeshCreator.cpp
    InputEventReader.cpp
    MeshGmshLoader.cpp
    MeshObjLoader.cpp
    MeshOffLoader.cpp
    MeshSTLLoader.cpp
    MeshTrianLoader.cpp
    MeshVTKLoader.cpp
    MeshXspLoader.cpp
    OffSequenceLoader.cpp
    ReadState.cpp
    ReadTopology.cpp
    SphereLoader.cpp
    StringMeshCreator.cpp
    VoxelGridLoader.cpp
    initLoader.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree SofaHelper)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_LOADER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
