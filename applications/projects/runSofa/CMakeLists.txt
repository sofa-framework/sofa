cmake_minimum_required(VERSION 2.8.12)
project(runSofa)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

add_executable(${PROJECT_NAME} ${RC_FILES} Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaComponentAdvanced SofaComponentMisc)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)

include(SofaPackageConfigHelpers)
sofa_install_targets(SofaGui runSofa "")
