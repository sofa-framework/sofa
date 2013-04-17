cmake_minimum_required(VERSION 2.8)

project("SofaLoader")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initLoader.h 
    loader/GridMeshCreator.h 
    loader/StringMeshCreator.h 
    loader/MeshGmshLoader.h 
    loader/MeshObjLoader.h 
    loader/MeshOffLoader.h 
    loader/MeshTrianLoader.h 
    loader/MeshVTKLoader.h 
    loader/MeshSTLLoader.h 
    loader/MeshXspLoader.h 
    loader/OffSequenceLoader.h 
    loader/SphereLoader.h 
    loader/VoxelGridLoader.h 
    misc/InputEventReader.h 
    misc/ReadState.h 
    misc/ReadState.inl 
    misc/ReadTopology.h 
    misc/ReadTopology.inl 

    )
    
set(SOURCE_FILES

    initLoader.cpp 
    loader/GridMeshCreator.cpp 
    loader/StringMeshCreator.cpp 
    loader/MeshGmshLoader.cpp 
    loader/MeshObjLoader.cpp 
    loader/MeshOffLoader.cpp 
    loader/MeshTrianLoader.cpp 
    loader/MeshVTKLoader.cpp 
    loader/MeshSTLLoader.cpp 
    loader/MeshXspLoader.cpp 
    loader/OffSequenceLoader.cpp 
    loader/SphereLoader.cpp 
    loader/VoxelGridLoader.cpp 
    misc/InputEventReader.cpp 
    misc/ReadState.cpp 
    misc/ReadTopology.cpp
 
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_LOADER" )
set(LINKER_DEPENDENCIES ${ZLIB_LIBRARIES_OPTIONAL} SofaSimulationTree )

    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
