cmake_minimum_required(VERSION 2.8)

project("SofaPhysicsBaseInteractiveModeler")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package(SofaPhysicsBasedInteractiveModelerLibs REQUIRED)
#include_directories(${SOFAPHYSICSBASEDINTERAVTIVEMODELER_INCLUDE_DIRS})

set(SOURCE_FILES

    initPim.cpp

	)

set(README_FILE PhysicsBasedInteractiveModeler.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PIM")
#set(LINKER_DEPENDENCIES SofaSimulationCommon SofaComponentLinearSolver SofaComponentOdeSolver SofaComponentMass SofaComponentForcefield SofaComponentCollision SofaComponentConstraint SofaComponentMisc SofaComponentBase SofaComponentMapping SofaComponentEngine SofaComponentVisualModel SofaPim SofaPimGuiQt SofaGuiQt)

include(${SOFA_CMAKE_DIR}/postProject.cmake)