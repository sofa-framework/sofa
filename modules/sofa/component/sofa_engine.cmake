cmake_minimum_required(VERSION 2.8)

project("SofaEngine")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initEngine.h \
    engine/AverageCoord.h \
    engine/AverageCoord.inl \
    engine/BoxROI.h \
    engine/BoxROI.inl \
    engine/PlaneROI.h \
    engine/PlaneROI.inl \
    engine/SphereROI.h \
    engine/SphereROI.inl \
    engine/DilateEngine.h \
    engine/DilateEngine.inl \
           engine/ExtrudeSurface.h \
           engine/ExtrudeSurface.inl \
           engine/ExtrudeQuadsAndGenerateHexas.h \
           engine/ExtrudeQuadsAndGenerateHexas.inl \
           engine/ExtrudeEdgesAndGenerateQuads.h \
           engine/ExtrudeEdgesAndGenerateQuads.inl \
           engine/GenerateRigidMass.h \
           engine/GenerateRigidMass.inl \
           engine/GroupFilterYoungModulus.h \
           engine/GroupFilterYoungModulus.inl \
           engine/MergeMeshes.h \
           engine/MergeMeshes.inl \
           engine/MergePoints.h \
           engine/MergePoints.inl \
           engine/MergeSets.h \
           engine/MergeSets.inl \
           engine/MeshBarycentricMapperEngine.h \
           engine/MeshBarycentricMapperEngine.inl \
	   engine/MeshROI.h \
	   engine/MeshROI.inl \
           engine/TransformPosition.h \
           engine/TransformPosition.inl \
           engine/TransformEngine.h \
           engine/TransformEngine.inl \
           engine/PointsFromIndices.h \
           engine/PointsFromIndices.inl \
           engine/ValuesFromIndices.h \
           engine/ValuesFromIndices.inl \
           engine/IndicesFromValues.h \
           engine/IndicesFromValues.inl \
           engine/IndexValueMapper.h \
           engine/IndexValueMapper.inl \
           engine/JoinPoints.h \
           engine/JoinPoints.inl \
           engine/MapIndices.h \
           engine/MapIndices.inl \
           engine/RandomPointDistributionInSurface.h \
           engine/RandomPointDistributionInSurface.inl \
           engine/Spiral.h \
           engine/Spiral.inl \
           engine/Vertex2Frame.h \
           engine/Vertex2Frame.inl \
           engine/TextureInterpolation.h \
           engine/TextureInterpolation.inl \
           engine/SubsetTopology.h \
           engine/SubsetTopology.inl \
           engine/RigidToQuatEngine.h \
           engine/RigidToQuatEngine.inl \
           engine/QuatToRigidEngine.h \
           engine/QuatToRigidEngine.inl \
           engine/ValuesFromPositions.h \
           engine/ValuesFromPositions.inl \
           engine/NormalsFromPoints.h \
           engine/NormalsFromPoints.inl \
           engine/ClusteringEngine.h \
           engine/ClusteringEngine.inl \
           engine/ShapeMatching.h \
           engine/ShapeMatching.inl \ 
           engine/ProximityROI.h \
           engine/ProximityROI.inl

    )
    
set(SOURCE_FILES


    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}  )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
