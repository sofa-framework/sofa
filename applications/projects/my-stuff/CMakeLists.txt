include(${SOFA_CMAKE_DIR}/preProject.cmake)

add_executable(${PROJECT_NAME} main.cpp)

AddLinkerDependencies(SofaSimulationTree SofaComponentMain SofaGuiMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

