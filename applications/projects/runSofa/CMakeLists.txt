cmake_minimum_required(VERSION 2.8.12)
project(runSofa)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

find_package(SofaFramework) # to get SOFA_HAVE_GLUT

add_executable(${PROJECT_NAME} ${RC_FILES} Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaComponentAdvanced SofaComponentMisc)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)

if(SOFA_HAVE_GLUT)
	target_compile_definitions(${PROJECT_NAME} PUBLIC "SOFA_HAVE_GLUT")
endif()

sofa_install_targets(SofaGui runSofa "")
