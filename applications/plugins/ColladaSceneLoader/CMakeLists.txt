include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	initPlugin.h
    SceneColladaLoader.h 
	)

set(SOURCE_FILES

	initPlugin.cpp
    SceneColladaLoader.cpp 
	)
	
set(README_FILES

	ColladaSceneLoader.txt
	)

include_directories("assimp/include")

if(WIN32)
	if(CMAKE_CL_64)
		link_directories("assimp/lib/x64")
	else()
		link_directories("assimp/lib/x86")
	endif()
	
	## assimp dll
	if(CMAKE_CL_64)
		file(GLOB sharedObjects "assimp/lib/x64/*.dll")
	else()
		file(GLOB sharedObjects "assimp/lib/x86/*.dll")
	endif()
	foreach(sharedObject ${sharedObjects})
		file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
	endforeach()
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_COLLADASCENELOADER")
AddLinkerDependencies(SofaLoader SofaSimulationCommon SofaBaseMechanics SofaRigid SofaBoundaryCondition SofaBaseTopology SofaBaseVisual SofaOpenglVisual assimp)

include(${SOFA_CMAKE_DIR}/postProject.cmake)