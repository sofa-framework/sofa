include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initComponentCommon.h)

set(SOURCE_FILES
    initComponentCommon.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_COMMON")
AddLinkerDependencies(Loader Rigid Deformable SimpleFem ObjectInteraction MeshCollision ExplicitOdeSolver ImplicitOdeSolver)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

