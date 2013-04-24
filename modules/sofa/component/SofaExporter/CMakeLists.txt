cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initExporter.h 
    ../misc/WriteState.h 
    ../misc/WriteState.inl 
    ../misc/WriteTopology.h 
    ../misc/WriteTopology.inl 
    ../misc/VTKExporter.h 
    ../misc/OBJExporter.h 
    ../misc/MeshExporter.h

    )
    
set(SOURCE_FILES

    ../initExporter.cpp 
    ../misc/WriteState.cpp 
    ../misc/WriteTopology.cpp 
    ../misc/VTKExporter.cpp 
    ../misc/OBJExporter.cpp 
    ../misc/MeshExporter.cpp
 
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_EXPORTER" )
set(LINKER_DEPENDENCIES ${ZLIB_LIBRARIES_OPTIONAL} SofaSimulationTree )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
