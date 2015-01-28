include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscTopology.h
    # ../misc/TopologicalChangeProcessor.h
    )
    
set(SOURCE_FILES

    ../initMiscTopology.cpp
    # ../misc/TopologicalChangeProcessor.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_TOPOLOGY;POINT_DATA_VECTOR_ACCESS")
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} FlowVR SofaBaseTopology )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
