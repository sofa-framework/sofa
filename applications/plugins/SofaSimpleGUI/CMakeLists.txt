include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

initSimpleGUI.h
SofaGLScene.h
Interactor.h
SpringInteractor.h
PickedPoint.h
SofaScene.h
SofaGL.h

	)

set(SOURCE_FILES

initSimpleGUI.cpp
SofaGLScene.cpp
Interactor.cpp
SpringInteractor.cpp
SofaScene.cpp
SofaGL.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFASIMPLEGUI")
AddLinkerDependencies(SofaSimulationGraph SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
