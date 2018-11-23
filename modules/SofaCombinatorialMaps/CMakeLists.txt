cmake_minimum_required(VERSION 3.1)
project(SofaCombinatorialMaps VERSION 1.0) 

# Dependencies

sofa_add_subdirectory_external(extlibs/CGoGN CGoGN ON)

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
		src/SofaCombinatorialMaps/Core/CMapTopology.h
		src/SofaCombinatorialMaps/Core/CMTopologyChange.h
		src/SofaCombinatorialMaps/Core/CMTopologyHandler.h
		src/SofaCombinatorialMaps/Core/CMTopologyElementHandler.h
		src/SofaCombinatorialMaps/Core/CMBaseTopologyData.h
		src/SofaCombinatorialMaps/Core/CMBaseTopologyEngine.h
		src/SofaCombinatorialMaps/Core/CMBaseTopology.h
	
	#BaseTopology
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyData.h
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyData.inl
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyDataHandler.h
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyDataHandler.inl
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyEngine.h
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyEngine.inl
		src/SofaCombinatorialMaps/BaseTopology/CMapTriangleSetTopologyContainer.h
		src/SofaCombinatorialMaps/BaseTopology/CMapTetrahedronSetTopologyContainer.h
		src/SofaCombinatorialMaps/BaseTopology/CMapHexahedronSetTopologyContainer.h
		src/SofaCombinatorialMaps/BaseTopology/SurfaceTopologyContainer.h
		src/SofaCombinatorialMaps/BaseTopology/SurfaceMaskTraversal.h
		src/SofaCombinatorialMaps/BaseTopology/VolumeTopologyContainer.h
		src/SofaCombinatorialMaps/BaseTopology/VolumeTopologyModifier.h
		src/SofaCombinatorialMaps/BaseTopology/VolumeGeometryAlgorithms.h
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyCoupling.h

	#ForceFields
		src/SofaCombinatorialMaps/ForceFields/CMHexahedronFEMForceField.h
		src/SofaCombinatorialMaps/ForceFields/CMHexahedronFEMForceField.inl
		src/SofaCombinatorialMaps/ForceFields/CMTetrahedralCorotationalFEMForceField.h
		src/SofaCombinatorialMaps/ForceFields/CMTetrahedralCorotationalFEMForceField.inl
		src/SofaCombinatorialMaps/ForceFields/CMTriangularFEMForceField.h
		src/SofaCombinatorialMaps/ForceFields/CMTriangularFEMForceField.inl
	)

list(APPEND SOURCE_FILES
	#core
		src/SofaCombinatorialMaps/Core/CMapTopology.cpp
		src/SofaCombinatorialMaps/Core/CMTopologyChange.cpp
		src/SofaCombinatorialMaps/Core/CMTopologyHandler.cpp
		src/SofaCombinatorialMaps/Core/CMTopologyElementHandler.cpp
		src/SofaCombinatorialMaps/Core/CMBaseTopology.cpp

	#BaseTopology
		src/SofaCombinatorialMaps/BaseTopology/CMapTriangleSetTopologyContainer.cpp
		src/SofaCombinatorialMaps/BaseTopology/CMapTetrahedronSetTopologyContainer.cpp
		src/SofaCombinatorialMaps/BaseTopology/CMapHexahedronSetTopologyContainer.cpp
		src/SofaCombinatorialMaps/BaseTopology/SurfaceTopologyContainer.cpp
		src/SofaCombinatorialMaps/BaseTopology/SurfaceMaskTraversal.cpp
		src/SofaCombinatorialMaps/BaseTopology/VolumeTopologyContainer.cpp
		src/SofaCombinatorialMaps/BaseTopology/VolumeTopologyModifier.cpp
		src/SofaCombinatorialMaps/BaseTopology/VolumeGeometryAlgorithms.cpp
		src/SofaCombinatorialMaps/BaseTopology/CMTopologyCoupling.cpp

	#ForceFields
		src/SofaCombinatorialMaps/ForceFields/CMHexahedronFEMForceField.cpp
		src/SofaCombinatorialMaps/ForceFields/CMTetrahedralCorotationalFEMForceField.cpp
		src/SofaCombinatorialMaps/ForceFields/CMTriangularFEMForceField.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaHelper SofaBaseTopology SofaOpenglVisual newmat ${cgogn_core_LIBRARIES} ${cgogn_geometry_LIBRARIES} ${cgogn_io_LIBRARIES} ${cgogn_modeling_LIBRARIES})

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFACOMBINATORIALMAPS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")

sofa_create_package(SofaCombinatorialMaps ${PROJECT_VERSION} SofaCombinatorialMaps SofaCombinatorialMaps)
