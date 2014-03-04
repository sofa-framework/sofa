include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initComponentMisc.h)

set(SOURCE_FILES
    initComponentMisc.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_MISC")
AddLinkerDependencies(MiscTopology MiscMapping MiscForceField MiscFem MiscEngine MiscCollision MiscSolver Misc)


include(${SOFA_CMAKE_DIR}/postProject.cmake)

