include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscEngine.h 
    # ../engine/Distances.h 
    # ../engine/Distances.inl
    # ../engine/ProjectiveTransformEngine.h
    # ../engine/ProjectiveTransformEngine.inl

    )
    
set(SOURCE_FILES

    ../initMiscEngine.cpp 
    # ../engine/Distances.cpp
    # ../engine/ProjectiveTransformEngine.cpp

    )
    

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_ENGINE" )
AddLinkerDependencies(SofaNonUniformFem )
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
