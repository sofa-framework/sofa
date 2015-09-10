include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES
    ManifoldEdgeSetGeometryAlgorithms.h
    ManifoldEdgeSetGeometryAlgorithms.inl
    ManifoldEdgeSetTopologyAlgorithms.h
    ManifoldEdgeSetTopologyAlgorithms.inl
    ManifoldEdgeSetTopologyContainer.h
    ManifoldEdgeSetTopologyModifier.h
    ManifoldTetrahedronSetTopologyContainer.h
    ManifoldTopologyObject_double.h
    ManifoldTopologyObject_float.h
    ManifoldTriangleSetTopologyAlgorithms.h
    ManifoldTriangleSetTopologyAlgorithms.inl
    ManifoldTriangleSetTopologyContainer.h
    ManifoldTriangleSetTopologyModifier.h
    config.h
)

set(SOURCE_FILES
    ManifoldEdgeSetGeometryAlgorithms.cpp
    ManifoldEdgeSetTopologyAlgorithms.cpp
    ManifoldEdgeSetTopologyContainer.cpp
    ManifoldEdgeSetTopologyModifier.cpp
    ManifoldTetrahedronSetTopologyContainer.cpp
    ManifoldTriangleSetTopologyAlgorithms.cpp
    ManifoldTriangleSetTopologyContainer.cpp
    ManifoldTriangleSetTopologyModifier.cpp
    initManifoldTopologies.cpp
)

set(README_FILES ManifoldTopologies.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_MANIFOLDTOPOLOGIES")
AddLinkerDependencies(SofaBaseTopology )

include(${SOFA_CMAKE_DIR}/postProject.cmake)