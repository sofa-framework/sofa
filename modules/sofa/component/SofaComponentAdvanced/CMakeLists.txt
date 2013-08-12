include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initComponentAdvanced.h
    
    )
    
set(SOURCE_FILES

    ../initComponentAdvanced.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_ADVANCED")
AddLinkerDependencies(SofaEulerianFluid SofaSphFluid SofaVolumetricData SofaNonUniformFem)
if(SOFA-EXTERNAL_HAVE_EIGEN2)
    AddLinkerDependencies(SofaEigen2Solver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
