include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initLoader.h 
    # ../loader/GridMeshCreator.h 
    # ../loader/StringMeshCreator.h 
    # ../loader/MeshGmshLoader.h 
    ../loader/MeshObjLoader.h 
	# ../loader/GIDMeshLoader.h
 #    ../loader/MeshOffLoader.h 
 #    ../loader/MeshTrianLoader.h 
    ../loader/MeshVTKLoader.h 
    # ../loader/MeshSTLLoader.h 
    # ../loader/MeshXspLoader.h 
    # ../loader/OffSequenceLoader.h 
    # ../loader/SphereLoader.h 
    # ../loader/VoxelGridLoader.h 
#     ../misc/InputEventReader.h
     ../misc/ReadState.h
     ../misc/ReadState.inl
    # ../misc/ReadTopology.h 
    # ../misc/ReadTopology.inl 
    )
    
set(SOURCE_FILES

    ../initLoader.cpp 
    # ../loader/GridMeshCreator.cpp 
    # ../loader/StringMeshCreator.cpp 
    # ../loader/MeshGmshLoader.cpp 
	../loader/MeshObjLoader.cpp
	# ../loader/GIDMeshLoader.cpp
 #    ../loader/MeshOffLoader.cpp 
 #    ../loader/MeshTrianLoader.cpp 
    ../loader/MeshVTKLoader.cpp 
    # ../loader/MeshSTLLoader.cpp 
    # ../loader/MeshXspLoader.cpp 
    # ../loader/OffSequenceLoader.cpp 
    # ../loader/SphereLoader.cpp 
    # ../loader/VoxelGridLoader.cpp 
    # ../misc/InputEventReader.cpp 
     ../misc/ReadState.cpp
    # ../misc/ReadTopology.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_LOADER" )
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} SofaSimulationTree tinyxml)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
