include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    customGUI.cpp
#    SimpleScene.cpp
)
set(HEADER_FILES
 #   SimpleScene.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimulationTree SofaComponentMain SofaGuiMain)
AddLinkerDependencies(SofaSimulationGraph)


include(${SOFA_CMAKE_DIR}/postProject.cmake)

