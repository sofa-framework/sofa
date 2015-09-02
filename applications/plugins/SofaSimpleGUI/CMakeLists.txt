include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    Camera.h
    Interactor.h
    PickedPoint.h
    SofaGL.h
    SofaScene.h
    SpringInteractor.h
    VisualPickVisitor.h
    config.h
)

set(SOURCE_FILES
    Camera.cpp
    Interactor.cpp
    SofaGL.cpp
    SofaScene.cpp
    SpringInteractor.cpp
    VisualPickVisitor.cpp
    initSimpleGUI.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES}  ${MOC_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFASIMPLEGUI")
AddLinkerDependencies(SofaSimulationGraph SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
