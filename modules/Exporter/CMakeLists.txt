include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initExporter.h
    WriteState.h
    WriteState.inl
    WriteTopology.h
    WriteTopology.inl
    VTKExporter.h
    OBJExporter.h
    STLExporter.h
    MeshExporter.h)

set(SOURCE_FILES
    initExporter.cpp
    WriteState.cpp
    WriteTopology.cpp
    VTKExporter.cpp
    OBJExporter.cpp
    STLExporter.cpp
    MeshExporter.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EXPORTER")
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

