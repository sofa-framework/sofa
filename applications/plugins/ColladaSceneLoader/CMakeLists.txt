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


# find assimp library & include directories...
if(APPLE)
	find_path(ASSIMP_INCLUDE_DIR NAMES assimp/Importer.hpp)
	include_directories(ASSIMP_INCLUDE_DIR)
else()
	include_directories("assimp/include")
endif()

if(WIN32)
	if(CMAKE_CL_64)
		link_directories("assimp/lib/x64")
	else()
		link_directories("assimp/lib/x86")
	endif()
	
	set(ASSIMP_LIBRARY assimp)
	
	## assimp dll
	if(CMAKE_CL_64)
		file(GLOB sharedObjects "assimp/lib/x64/*.dll")
	else()
		file(GLOB sharedObjects "assimp/lib/x86/*.dll")
	endif()
	foreach(sharedObject ${sharedObjects})
		file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
	endforeach()
else()
	find_library(ASSIMP_LIBRARY NAMES assimp)
    #todo add a test for assimp version >= 3
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_COLLADASCENELOADER")
AddLinkerDependencies(SofaLoader SofaSimulationCommon SofaBaseMechanics SofaRigid SofaBoundaryCondition SofaBaseTopology SofaBaseVisual SofaOpenglVisual SofaDeformable SofaMeshCollision ${ASSIMP_LIBRARY})

if(SOFA-PLUGIN_FLEXIBLE)
	AddLinkerDependencies(Flexible)
endif()

if(SOFA-PLUGIN_IMAGE)
    AddLinkerDependencies(image)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
