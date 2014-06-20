include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    ManifoldTopologies.h
    ManifoldEdgeSetGeometryAlgorithms.h
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
    ManifoldTopologyObject_float.h)

set(SOURCE_FILES
    ManifoldTopologies.cpp
    ManifoldEdgeSetGeometryAlgorithms.cpp
    ManifoldEdgeSetTopologyAlgorithms.cpp
    ManifoldEdgeSetTopologyContainer.cpp
    ManifoldEdgeSetTopologyModifier.cpp
    ManifoldTriangleSetTopologyContainer.cpp
    ManifoldTriangleSetTopologyModifier.cpp
    ManifoldTriangleSetTopologyAlgorithms.cpp
    ManifoldTetrahedronSetTopologyContainer.cpp)

set(README_FILES ManifoldTopologies.txt)

sofa_add_plugin(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
AddCompilerDefinitions("SOFA_BUILD_MANIFOLDTOPOLOGIES")
AddLinkerDependencies(SofaBaseTopology)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
