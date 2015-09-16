cmake_minimum_required(VERSION 2.8.12)
project(ManifoldTopologies)

set(MANIFOLDTOPOLOGIES_VERSION 0.1)

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

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${MANIFOLDTOPOLOGIES_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MANIFOLD_TOPOLOGIES")
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)

install(TARGETS ${PROJECT_NAME}
        COMPONENT ManifoldTopologies_libraries
        EXPORT ManifoldTopologiesTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
