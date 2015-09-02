include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    PreassembledMass.h
    PreassembledMass.inl
    config.h
)

set(SOURCE_FILES
    PreassembledMass.cpp
    initPlugin.cpp
)

if(SOFA-PLUGIN_FLEXIBLE)

    AddLinkerDependencies(Flexible)
    AddCompilerDefinitions("SOFA_HAVE_FLEXIBLE" )

endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PreassembledMass")
AddLinkerDependencies(Compliant)


include(${SOFA_CMAKE_DIR}/postProject.cmake)
