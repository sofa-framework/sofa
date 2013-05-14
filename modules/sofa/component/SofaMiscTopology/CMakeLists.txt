include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscTopology.h
    ../misc/TopologicalChangeProcessor.h
    )
    
set(SOURCE_FILES

    ../initMiscTopology.cpp
    ../misc/TopologicalChangeProcessor.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_MISC_TOPOLOGY;POINT_DATA_VECTOR_ACCESS")
set(LINKER_DEPENDENCIES ${ZLIB_LIBRARIES_OPTIONAL} miniFlowVR SofaBaseTopology )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
