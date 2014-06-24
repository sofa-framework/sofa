include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initMiscEngine.h
    Distances.h
    Distances.inl
    ProjectiveTransformEngine.h
    ProjectiveTransformEngine.inl

    )

set(SOURCE_FILES

    initMiscEngine.cpp
    Distances.cpp
    ProjectiveTransformEngine.cpp

    )


set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_ENGINE" )
AddLinkerDependencies(SofaNonUniformFem )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
