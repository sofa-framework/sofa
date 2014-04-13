include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    customGUI.cpp
    Interactor.cpp
    SpringInteractor.cpp
    SofaGlInterface.cpp
)
set(HEADER_FILES
    Interactor.h
    SpringInteractor.h
    PickedPoint.h
    SofaGlInterface.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimulationTree SofaComponentMain SofaGuiMain)
AddLinkerDependencies(SofaSimulationGraph)
AddLinkerDependencies(Compliant)


include(${SOFA_CMAKE_DIR}/postProject.cmake)

