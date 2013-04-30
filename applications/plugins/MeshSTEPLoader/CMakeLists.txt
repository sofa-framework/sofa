cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package(SofaMeshStepLooaderLibs REQUIRED)
#include_directories(${SOFAMESHSTEPLOADER_INCLUDE_DIRS})

set(HEADER_FILES

    MeshSTEPLoader.h
    ParametricTriangleTopologyContainer.h 
    SingleComponent.inl
    SingleComponent.h 
    STEPShapeMapping.h 
    initMeshSTEPLoader.h

	)

set(SOURCE_FILES

    MeshSTEPLoader.cpp 
    ParametricTriangleTopologyContainer.cpp 
    SingleComponent.cpp 
    STEPShapeMapping.cpp 
    initMeshSTEPLoader.cpp

	)

set(README_FILES

	PluginMeshSTEPLoader.txt
	)	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_MESHSTEPLOADERPLUGIN")
set(LINKER_DEPENDENCIES SofaBaseTopology )

if(UNIX)
    list(APPEND LINKER_DEPENDENCIES "TK")
    list(APPEND LINKER_DEPENDENCIES "GL")
endif()

if(WIN32)
    list(APPEND LINKER_DEPENDENCIES "OpenCascade" )
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)