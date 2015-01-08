include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initComponentMisc.h

    )
    
set(SOURCE_FILES

    ../initComponentMisc.cpp
 
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} ${SOFA_LIB_TYPE} ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_MISC" )
AddLinkerDependencies(SofaMiscTopology SofaMiscMapping SofaMiscForceField SofaMiscFem SofaMiscEngine SofaMiscCollision SofaMiscSolver SofaMisc )

    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
