include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initSceneCreator.h
    SceneCreator.h
        GetVectorVisitor.h
        GetAssembledSizeVisitor.h

	)

set(SOURCE_FILES

    initSceneCreator.cpp
    SceneCreator.cpp
        GetVectorVisitor.cpp
        GetAssembledSizeVisitor.cpp

	)

AddCompilerDefinitions("SOFA_BUILD_SCENECREATOR")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
AddLinkerDependencies(
    SofaSimulationTree
    SofaComponentMain
    SofaSimulationGraph
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
