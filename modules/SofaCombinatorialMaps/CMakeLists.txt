cmake_minimum_required(VERSION 3.1)
project(SofaCombinatorialMaps VERSION 1.0) 

# Dependencies

 
add_subdirectory(cgogn2)
#list(APPEND CGOGN_TARGETS cgogn_core cgogn_geometry cgogn_io cgogn_modeling)

find_package(Newmat)
find_package(cgogn_core REQUIRED)
find_package(cgogn_geometry REQUIRED)
find_package(cgogn_io REQUIRED)
find_package(cgogn_modeling REQUIRED)

# Config
set(HEADER_FILES
        src/SofaCombinatorialMaps/config.h
	)
set(SOURCE_FILES
        src/SofaCombinatorialMaps/initCombinatorialMaps.cpp
	)


 # Sources
list(APPEND HEADER_FILES
	#core
        src/SofaCombinatorialMaps/MapTopology.h
        src/SofaCombinatorialMaps/CMTopologyChange.h
        src/SofaCombinatorialMaps/CMTopologyHandler.h
        src/SofaCombinatorialMaps/CMTopologyElementHandler.h
        src/SofaCombinatorialMaps/CMBaseTopologyData.h
        src/SofaCombinatorialMaps/CMBaseTopologyEngine.h
        src/SofaCombinatorialMaps/CMBaseTopology.h
	
	#BaseTopology
        src/SofaCombinatorialMaps/CMTopologyData.h
        src/SofaCombinatorialMaps/CMTopologyData.inl
        src/SofaCombinatorialMaps/CMTopologyDataHandler.h
        src/SofaCombinatorialMaps/CMTopologyDataHandler.inl
        src/SofaCombinatorialMaps/CMTopologyEngine.h
        src/SofaCombinatorialMaps/CMTopologyEngine.inl
        src/SofaCombinatorialMaps/MapTriangleSetTopologyContainer.h
        src/SofaCombinatorialMaps/MapTetrahedronSetTopologyContainer.h
        src/SofaCombinatorialMaps/MapHexahedronSetTopologyContainer.h
        src/SofaCombinatorialMaps/SurfaceTopologyContainer.h
        src/SofaCombinatorialMaps/SurfaceMaskTraversal.h
        src/SofaCombinatorialMaps/VolumeTopologyContainer.h
        src/SofaCombinatorialMaps/VolumeTopologyModifier.h
        src/SofaCombinatorialMaps/VolumeGeometryAlgorithms.h
        src/SofaCombinatorialMaps/CMTopologyCoupling.h
        src/SofaCombinatorialMaps/CMTopologyCoupling.inl

	#ForceFields
        src/SofaCombinatorialMaps/CMHexahedronFEMForceField.h
        src/SofaCombinatorialMaps/CMHexahedronFEMForceField.inl
        src/SofaCombinatorialMaps/CMTetrahedralCorotationalFEMForceField.h
        src/SofaCombinatorialMaps/CMTetrahedralCorotationalFEMForceField.inl
        src/SofaCombinatorialMaps/CMTriangularFEMForceField.h
        src/SofaCombinatorialMaps/CMTriangularFEMForceField.inl
	)

list(APPEND SOURCE_FILES
	#core
        src/SofaCombinatorialMaps/MapTopology.cpp
        src/SofaCombinatorialMaps/CMTopologyChange.cpp
        src/SofaCombinatorialMaps/CMTopologyHandler.cpp
        src/SofaCombinatorialMaps/CMTopologyElementHandler.cpp
        src/SofaCombinatorialMaps/CMBaseTopology.cpp

	#BaseTopology
        src/SofaCombinatorialMaps/MapTriangleSetTopologyContainer.cpp
        src/SofaCombinatorialMaps/MapTetrahedronSetTopologyContainer.cpp
        src/SofaCombinatorialMaps/MapHexahedronSetTopologyContainer.cpp
        src/SofaCombinatorialMaps/SurfaceTopologyContainer.cpp
        src/SofaCombinatorialMaps/SurfaceMaskTraversal.cpp
        src/SofaCombinatorialMaps/VolumeTopologyContainer.cpp
        src/SofaCombinatorialMaps/VolumeTopologyModifier.cpp
        src/SofaCombinatorialMaps/VolumeGeometryAlgorithms.cpp
        src/SofaCombinatorialMaps/CMTopologyCoupling.cpp

	#ForceFields
        src/SofaCombinatorialMaps/CMHexahedronFEMForceField.cpp
        src/SofaCombinatorialMaps/CMTetrahedralCorotationalFEMForceField.cpp
        src/SofaCombinatorialMaps/CMTriangularFEMForceField.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaHelper SofaBaseTopology newmat ${cgogn_core_LIBRARIES} ${cgogn_geometry_LIBRARIES} ${cgogn_io_LIBRARIES} ${cgogn_modeling_LIBRARIES})

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFACOMBINATORIALMAPS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")

sofa_create_package(SofaCombinatorialMaps ${PROJECT_VERSION} SofaCombinatorialMaps SofaCombinatorialMaps)
