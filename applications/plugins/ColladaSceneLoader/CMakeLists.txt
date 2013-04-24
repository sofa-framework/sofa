cmake_minimum_required(VERSION 2.8)

project("ColladaSceneLoader")

include(${SOFA_CMAKE_DIR}/preProject.cmake)


##plugin external dependencies
find_package(ColladaSceneLoaderLibs REQUIRED)
include_directories(${COLLADASCENELOADER_INCLUDE_DIRS})

set(HEADER_FILES

    SceneColladaLoader.h 
	initPlugin.h

	)

set(SOURCE_FILES

    SceneColladaLoader.cpp 
	initPlugin.cpp

	)
	
set(README_FILE ColladaSceneLoader.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_COLLADASCENELOADER")
set(LINKER_DEPENDENCIES SofaLoader SofaSimulationCommon SofaBaseMechanics SofaRigid SofaBoundaryCondition SofaBaseTopology SofaBaseVisual SofaOpenglVisual assimp)

include(${SOFA_CMAKE_DIR}/postProject.cmake)