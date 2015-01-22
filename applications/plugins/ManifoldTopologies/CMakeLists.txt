cmake_minimum_required(VERSION 2.8)
project(ManifoldTopologies)

set(MANIFOLDTOPOLOGIES_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(MANIFOLDTOPOLOGIES_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(MANIFOLDTOPOLOGIES_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
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
    ManifoldTopologyObject_float.h)

set(SOURCE_FILES
    initManifoldTopologies.cpp
    ManifoldEdgeSetGeometryAlgorithms.cpp
    ManifoldEdgeSetTopologyAlgorithms.cpp
    ManifoldEdgeSetTopologyContainer.cpp
    ManifoldEdgeSetTopologyModifier.cpp
    ManifoldTriangleSetTopologyContainer.cpp
    ManifoldTriangleSetTopologyModifier.cpp
    ManifoldTriangleSetTopologyAlgorithms.cpp
    ManifoldTetrahedronSetTopologyContainer.cpp)

set(README_FILES
	ManifoldTopologies.txt)

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MANIFOLDTOPOLOGIES")
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)

install(TARGETS ${PROJECT_NAME}
    COMPONENT ManifoldTopologies_libraries
    EXPORT ManifoldTopologiesTargets
    RUNTIME DESTINATION "${MANIFOLDTOPOLOGIES_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${MANIFOLDTOPOLOGIES_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${MANIFOLDTOPOLOGIES_INSTALL_LIB_DIR}")
