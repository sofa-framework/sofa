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

AddCompilerDefinitions("SOFA_BUILD_MESHSTEPLOADERPLUGIN")
AddLinkerDependencies(SofaBaseTopology )

if(UNIX)
    AddLinkerDependencies("TK")
    AddLinkerDependencies("GL")
endif()

if(WIN32)
    AddLinkerDependencies("OpenCascade" )
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)