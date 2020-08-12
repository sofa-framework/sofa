cmake_minimum_required(VERSION 3.12)
project(SofaCombinatorialMaps VERSION 1.0) 

# Dependencies
set(CGOGN_BUILD_TESTS OFF)
set(CGOGN_BUILD_EXAMPLES OFF)
set(CGOGN_BUILD_BENCHS OFF)
set(CGOGN_USE_QT OFF)
sofa_add_subdirectory_external(extlibs/CGoGN CGoGN ON)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/lib/cmake")

find_package(Newmat REQUIRED)
find_package(cgogn_core REQUIRED)
find_package(cgogn_geometry REQUIRED)
find_package(cgogn_io REQUIRED)
find_package(cgogn_modeling REQUIRED)

set(SRC_ROOT src/SofaCombinatorialMaps)
# Config
set(HEADER_FILES
    ${SRC_ROOT}/config.h
    )
set(SOURCE_FILES
    ${SRC_ROOT}/initCombinatorialMaps.cpp
    )
# Sources
list(APPEND HEADER_FILES
    #core
    ${SRC_ROOT}/Core/CMapTopology.h
    ${SRC_ROOT}/Core/CMTopologyChange.h
    ${SRC_ROOT}/Core/CMTopologyHandler.h
    ${SRC_ROOT}/Core/CMTopologyElementHandler.h
    ${SRC_ROOT}/Core/CMBaseTopologyData.h
    ${SRC_ROOT}/Core/CMBaseTopologyEngine.h
    ${SRC_ROOT}/Core/CMBaseTopology.h
    
    #BaseTopology
    ${SRC_ROOT}/BaseTopology/CMTopologyData.h
    ${SRC_ROOT}/BaseTopology/CMTopologyData.inl
    ${SRC_ROOT}/BaseTopology/CMTopologyDataHandler.h
    ${SRC_ROOT}/BaseTopology/CMTopologyDataHandler.inl
    ${SRC_ROOT}/BaseTopology/CMTopologyEngine.h
    ${SRC_ROOT}/BaseTopology/CMTopologyEngine.inl
    ${SRC_ROOT}/BaseTopology/CMapTriangleSetTopologyContainer.h
    ${SRC_ROOT}/BaseTopology/CMapTetrahedronSetTopologyContainer.h
    ${SRC_ROOT}/BaseTopology/CMapHexahedronSetTopologyContainer.h
    ${SRC_ROOT}/BaseTopology/SurfaceTopologyContainer.h
    ${SRC_ROOT}/BaseTopology/SurfaceMaskTraversal.h
    ${SRC_ROOT}/BaseTopology/VolumeTopologyContainer.h
    ${SRC_ROOT}/BaseTopology/VolumeTopologyModifier.h
    ${SRC_ROOT}/BaseTopology/VolumeGeometryAlgorithms.h
    ${SRC_ROOT}/BaseTopology/CMTopologyCoupling.h

    #ForceFields
    ${SRC_ROOT}/ForceFields/CMHexahedronFEMForceField.h
    ${SRC_ROOT}/ForceFields/CMHexahedronFEMForceField.inl
    ${SRC_ROOT}/ForceFields/CMTetrahedralCorotationalFEMForceField.h
    ${SRC_ROOT}/ForceFields/CMTetrahedralCorotationalFEMForceField.inl
    ${SRC_ROOT}/ForceFields/CMTriangularFEMForceField.h
    ${SRC_ROOT}/ForceFields/CMTriangularFEMForceField.inl
    )
list(APPEND SOURCE_FILES
    #core
    ${SRC_ROOT}/Core/CMapTopology.cpp
    ${SRC_ROOT}/Core/CMTopologyChange.cpp
    ${SRC_ROOT}/Core/CMTopologyHandler.cpp
    ${SRC_ROOT}/Core/CMTopologyElementHandler.cpp
    ${SRC_ROOT}/Core/CMBaseTopology.cpp

    #BaseTopology
    ${SRC_ROOT}/BaseTopology/CMapTriangleSetTopologyContainer.cpp
    ${SRC_ROOT}/BaseTopology/CMapTetrahedronSetTopologyContainer.cpp
    ${SRC_ROOT}/BaseTopology/CMapHexahedronSetTopologyContainer.cpp
    ${SRC_ROOT}/BaseTopology/SurfaceTopologyContainer.cpp
    ${SRC_ROOT}/BaseTopology/SurfaceMaskTraversal.cpp
    ${SRC_ROOT}/BaseTopology/VolumeTopologyContainer.cpp
    ${SRC_ROOT}/BaseTopology/VolumeTopologyModifier.cpp
    ${SRC_ROOT}/BaseTopology/VolumeGeometryAlgorithms.cpp
    ${SRC_ROOT}/BaseTopology/CMTopologyCoupling.cpp

    #ForceFields
    ${SRC_ROOT}/ForceFields/CMHexahedronFEMForceField.cpp
    ${SRC_ROOT}/ForceFields/CMTetrahedralCorotationalFEMForceField.cpp
    ${SRC_ROOT}/ForceFields/CMTriangularFEMForceField.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaHelper SofaBaseTopology SofaOpenglVisual)
target_link_libraries(${PROJECT_NAME} PUBLIC newmat ${cgogn_core_LIBRARIES} ${cgogn_geometry_LIBRARIES} ${cgogn_io_LIBRARIES} ${cgogn_modeling_LIBRARIES})

sofa_create_package_with_targets(
    PACKAGE_NAME SofaCombinatorialMaps 
    PACKAGE_VERSION ${PROJECT_VERSION} 
    TARGETS SofaCombinatorialMaps AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR src
    INCLUDE_INSTALL_DIR SofaCombinatorialMaps
    RELOCATABLE "plugins"
    )
