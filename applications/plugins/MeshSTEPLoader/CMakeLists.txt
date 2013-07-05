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

find_package(OPENCASCADE REQUIRED)
include_directories(${OPENCASCADE_INCLUDE_DIR})
#required for OpenCASCADE
AddCompilerDefinitions("WNT")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_MESHSTEPLOADERPLUGIN")
AddLinkerDependencies(SofaBaseTopology )

set(TKLibraries

    TKBRep
    TKernel
    TKG3d
    TKGeomBase
    TKMath
    TKMesh
    TKPrim
    TKShHealing
    TKStep
    TKStepBase
    TKTopAlgo
    TKXSBase
)

foreach(TKLibrary ${TKLibraries})
    AddLinkerDependencies(${OPENCASCADE_LIBRARIES_DIR}/${TKLibrary}.lib)
    file(COPY ${OPENCASCADE_LIBRARIES_DIR}/../bin/${TKLibrary}.dll DESTINATION ${SOFA_BIN_DIR})
endforeach()

include(${SOFA_CMAKE_DIR}/postProject.cmake)