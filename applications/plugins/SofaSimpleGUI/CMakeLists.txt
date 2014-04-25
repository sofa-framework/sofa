include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initSimpleGUI.h SofaGlInterface.h Interactor.h SpringInteractor.h PickedPoint.h

	)

set(SOURCE_FILES

    initSimpleGUI.cpp SofaGlInterface.cpp Interactor.cpp SpringInteractor.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
AddLinkerDependencies(SofaSimulationGraph SofaComponentMain SofaGuiMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
