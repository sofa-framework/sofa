include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    customGUI.cpp
    Anchor.cpp
    DragAnchor.cpp
    GlPickedPoint.cpp
)
set(HEADER_FILES
    Anchor.h
    DragAnchor.h
    GlPickedPoint.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimulationTree SofaComponentMain SofaGuiMain)
AddLinkerDependencies(SofaSimulationGraph)
AddLinkerDependencies(Compliant)


include(${SOFA_CMAKE_DIR}/postProject.cmake)

