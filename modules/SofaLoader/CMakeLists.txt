include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initLoader.h
    GridMeshCreator.h
    StringMeshCreator.h
    MeshGmshLoader.h
    MeshObjLoader.h
	GIDMeshLoader.h
    MeshOffLoader.h
    MeshTrianLoader.h
    MeshVTKLoader.h
    MeshSTLLoader.h
    MeshXspLoader.h
    OffSequenceLoader.h
    SphereLoader.h
    VoxelGridLoader.h
    InputEventReader.h
    ReadState.h
    ReadState.inl
    ReadTopology.h
    ReadTopology.inl
    )

set(SOURCE_FILES

    initLoader.cpp
    GridMeshCreator.cpp
    StringMeshCreator.cpp
    MeshGmshLoader.cpp
	MeshObjLoader.cpp
	GIDMeshLoader.cpp
    MeshOffLoader.cpp
    MeshTrianLoader.cpp
    MeshVTKLoader.cpp
    MeshSTLLoader.cpp
    MeshXspLoader.cpp
    OffSequenceLoader.cpp
    SphereLoader.cpp
    VoxelGridLoader.cpp
    InputEventReader.cpp
    ReadState.cpp
    ReadTopology.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_LOADER" )
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} SofaSimulationTree tinyxml)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
