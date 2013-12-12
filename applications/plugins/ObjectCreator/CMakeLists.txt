include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initSceneCreator.h
    SceneCreator.h

	)

set(SOURCE_FILES

    initSceneCreator.cpp
    SceneCreator.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
AddLinkerDependencies(SofaSimulationTree SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
