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

if(SOFA-LIB_GUI_QT)
    UseQt()
    set(SOURCE_FILES  ${SOURCE_FILES}
        QSofaScene.cpp
    )
    set(HEADER_FILES ${HEADER_FILES}
        QSofaScene.h
    )
    SOFA_QT4_WRAP_CPP(MOC_FILES QSofaScene.h)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES}  ${MOC_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFASIMPLEGUI")
AddLinkerDependencies(SofaSimulationGraph SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
