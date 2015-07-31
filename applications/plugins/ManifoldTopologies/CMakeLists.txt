include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    config.h
    ManifoldEdgeSetGeometryAlgorithms.h 
    ManifoldEdgeSetGeometryAlgorithms.inl 
    ManifoldEdgeSetTopologyAlgorithms.h 
    ManifoldEdgeSetTopologyAlgorithms.inl 
    ManifoldEdgeSetTopologyContainer.h 
    ManifoldEdgeSetTopologyModifier.h 
    ManifoldTriangleSetTopologyContainer.h 
    ManifoldTriangleSetTopologyModifier.h 
    ManifoldTriangleSetTopologyAlgorithms.h 
    ManifoldTriangleSetTopologyAlgorithms.inl 
    ManifoldTetrahedronSetTopologyContainer.h 
    ManifoldTopologyObject_double.h 
    ManifoldTopologyObject_float.h

	)

set(SOURCE_FILES

    initManifoldTopologies.cpp 
    ManifoldEdgeSetGeometryAlgorithms.cpp 
    ManifoldEdgeSetTopologyAlgorithms.cpp 
    ManifoldEdgeSetTopologyContainer.cpp 
    ManifoldEdgeSetTopologyModifier.cpp 
    ManifoldTriangleSetTopologyContainer.cpp 
    ManifoldTriangleSetTopologyModifier.cpp 
    ManifoldTriangleSetTopologyAlgorithms.cpp 
    ManifoldTetrahedronSetTopologyContainer.cpp

	)

set(README_FILES

	ManifoldTopologies.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_MANIFOLDTOPOLOGIES")
AddLinkerDependencies(SofaBaseTopology )

include(${SOFA_CMAKE_DIR}/postProject.cmake)