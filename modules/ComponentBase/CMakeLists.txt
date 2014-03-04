include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initComponentBase.h)

set(SOURCE_FILES
    initComponentBase.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_BASE")
AddLinkerDependencies(BaseTopology BaseMechanics BaseCollision BaseLinearSolver BaseAnimationLoop BaseVisual)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

