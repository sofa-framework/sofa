include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    THMPGHashTable.h
    THMPGSpatialHashing.h
    config.h
)

set(SOURCE_FILES
    THMPGHashTable.cpp
    THMPGSpatialHashing.cpp
    initTHMPGSpatialHashingPlugin.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
AddCompilerDefinitions("SOFA_BUILD_THMPGSPATIALHASHING")
AddLinkerDependencies(SofaBaseCollision)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
