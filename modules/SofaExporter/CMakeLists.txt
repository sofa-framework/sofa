include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initExporter.h
    WriteState.h
    WriteState.inl
    WriteTopology.h
    WriteTopology.inl
    VTKExporter.h
    OBJExporter.h
    STLExporter.h
    MeshExporter.h
    BlenderExporter.h
    BlenderExporter.inl

    )

set(SOURCE_FILES

    initExporter.cpp
    WriteState.cpp
    WriteTopology.cpp
    VTKExporter.cpp
    OBJExporter.cpp
    STLExporter.cpp
    MeshExporter.cpp
    BlenderExporter.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EXPORTER" )
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} SofaSimulationTree )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
