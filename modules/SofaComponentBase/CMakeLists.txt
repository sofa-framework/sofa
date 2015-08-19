include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initComponentBase.h

    )

set(SOURCE_FILES

    initComponentBase.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_BASE")
AddLinkerDependencies(SofaBaseTopology SofaBaseMechanics SofaBaseCollision SofaBaseLinearSolver SofaBaseAnimationLoop SofaBaseVisual)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
