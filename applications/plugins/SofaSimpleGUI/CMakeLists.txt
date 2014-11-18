include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

initSimpleGUI.h
Camera.h
Interactor.h
SpringInteractor.h
PickedPoint.h
SofaScene.h
SofaGL.h
VisualPickVisitor.h

	)

set(SOURCE_FILES

initSimpleGUI.cpp
Camera.cpp
Interactor.cpp
SpringInteractor.cpp
SofaScene.cpp
SofaGL.cpp
VisualPickVisitor.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES}  ${MOC_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFASIMPLEGUI")
AddLinkerDependencies(SofaSimulationGraph SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
