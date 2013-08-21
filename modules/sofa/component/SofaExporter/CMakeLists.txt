include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initExporter.h 
    ../misc/WriteState.h 
    ../misc/WriteState.inl 
    ../misc/WriteTopology.h 
    ../misc/WriteTopology.inl 
    ../misc/VTKExporter.h 
    ../misc/OBJExporter.h 
    ../misc/MDDExporter.h 
    ../misc/INPExporter.h 
    ../misc/INPExporterMaster.h 
    ../misc/STLExporter.h 
    ../misc/MeshExporter.h

    )
    
set(SOURCE_FILES

    ../initExporter.cpp 
    ../misc/WriteState.cpp 
    ../misc/WriteTopology.cpp 
    ../misc/VTKExporter.cpp 
    ../misc/OBJExporter.cpp 
    ../misc/MDDExporter.cpp 
    ../misc/INPExporter.cpp 
    ../misc/INPExporterMaster.cpp 
    ../misc/STLExporter.cpp 
    ../misc/MeshExporter.cpp
 
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EXPORTER" )
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} SofaSimulationTree )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
