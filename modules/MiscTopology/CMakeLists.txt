include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initMiscTopology.h
    TopologicalChangeProcessor.h)

set(SOURCE_FILES
    initMiscTopology.cpp
    TopologicalChangeProcessor.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_TOPOLOGY;POINT_DATA_VECTOR_ACCESS")
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} FlowVR BaseTopology)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

