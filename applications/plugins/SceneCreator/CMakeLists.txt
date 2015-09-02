include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    GetAssembledSizeVisitor.h
    GetVectorVisitor.h
    SceneCreator.h
    config.h
)

set(SOURCE_FILES
    GetAssembledSizeVisitor.cpp
    GetVectorVisitor.cpp
    SceneCreator.cpp
    initSceneCreator.cpp
)

AddCompilerDefinitions("SOFA_BUILD_SCENECREATOR")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
AddLinkerDependencies(
    SofaSimulationTree
    SofaComponentMain
    SofaSimulationGraph
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
