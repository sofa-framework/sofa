cmake_minimum_required(VERSION 2.8.12)
project(runSofa)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

find_package(SofaFramework) # to get SOFA_HAVE_GLUT

if(APPLE)
	add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${RC_FILES} Main.cpp)
else()
	add_executable(${PROJECT_NAME} ${RC_FILES} Main.cpp)
endif()

target_link_libraries(${PROJECT_NAME} SofaComponentAdvanced SofaComponentMisc)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)

if(SOFA_HAVE_GLUT)
	target_compile_definitions(${PROJECT_NAME} PUBLIC "SOFA_HAVE_GLUT_GUI")
endif()

if(APPLE)
	include(cmake/bundle.cmake)
else()
	sofa_install_targets(SofaGui runSofa "")
endif()