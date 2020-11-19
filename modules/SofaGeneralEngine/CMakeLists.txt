cmake_minimum_required(VERSION 3.12)
project(SofaGeneralEngine LANGUAGES CXX)


set(SOFAGENERALENGINE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/config.h.in
    ${SOFAGENERALENGINE_SRC}/initSofaGeneralEngine.h
    )

set(SOURCE_FILES
    ${SOFAGENERALENGINE_SRC}/initSofaGeneralEngine.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/AverageCoord.h
    ${SOFAGENERALENGINE_SRC}/AverageCoord.inl
    ${SOFAGENERALENGINE_SRC}/ClusteringEngine.h
    ${SOFAGENERALENGINE_SRC}/ClusteringEngine.inl
    ${SOFAGENERALENGINE_SRC}/ComplementaryROI.h
    ${SOFAGENERALENGINE_SRC}/ComplementaryROI.inl
    ${SOFAGENERALENGINE_SRC}/DilateEngine.h
    ${SOFAGENERALENGINE_SRC}/DilateEngine.inl
    ${SOFAGENERALENGINE_SRC}/DifferenceEngine.h
    ${SOFAGENERALENGINE_SRC}/DifferenceEngine.inl
    ${SOFAGENERALENGINE_SRC}/ExtrudeEdgesAndGenerateQuads.h
    ${SOFAGENERALENGINE_SRC}/ExtrudeEdgesAndGenerateQuads.inl
    ${SOFAGENERALENGINE_SRC}/ExtrudeQuadsAndGenerateHexas.h
    ${SOFAGENERALENGINE_SRC}/ExtrudeQuadsAndGenerateHexas.inl
    ${SOFAGENERALENGINE_SRC}/ExtrudeSurface.h
    ${SOFAGENERALENGINE_SRC}/ExtrudeSurface.inl
    ${SOFAGENERALENGINE_SRC}/GenerateCylinder.h
    ${SOFAGENERALENGINE_SRC}/GenerateCylinder.inl
    ${SOFAGENERALENGINE_SRC}/GenerateGrid.h
    ${SOFAGENERALENGINE_SRC}/GenerateGrid.inl
    ${SOFAGENERALENGINE_SRC}/GenerateRigidMass.h
    ${SOFAGENERALENGINE_SRC}/GenerateRigidMass.inl
    ${SOFAGENERALENGINE_SRC}/GenerateSphere.h
    ${SOFAGENERALENGINE_SRC}/GenerateSphere.inl
    ${SOFAGENERALENGINE_SRC}/GroupFilterYoungModulus.h
    ${SOFAGENERALENGINE_SRC}/GroupFilterYoungModulus.inl
    ${SOFAGENERALENGINE_SRC}/HausdorffDistance.h
    ${SOFAGENERALENGINE_SRC}/HausdorffDistance.inl
    ${SOFAGENERALENGINE_SRC}/IndexValueMapper.h
    ${SOFAGENERALENGINE_SRC}/IndexValueMapper.inl
    ${SOFAGENERALENGINE_SRC}/Indices2ValuesMapper.h
    ${SOFAGENERALENGINE_SRC}/Indices2ValuesMapper.inl
    ${SOFAGENERALENGINE_SRC}/IndicesFromValues.h
    ${SOFAGENERALENGINE_SRC}/IndicesFromValues.inl
    ${SOFAGENERALENGINE_SRC}/JoinPoints.h
    ${SOFAGENERALENGINE_SRC}/JoinPoints.inl
    ${SOFAGENERALENGINE_SRC}/MapIndices.h
    ${SOFAGENERALENGINE_SRC}/MapIndices.inl
    ${SOFAGENERALENGINE_SRC}/MathOp.h
    ${SOFAGENERALENGINE_SRC}/MathOp.inl
    ${SOFAGENERALENGINE_SRC}/MergeMeshes.h
    ${SOFAGENERALENGINE_SRC}/MergeMeshes.inl
    ${SOFAGENERALENGINE_SRC}/MergePoints.h
    ${SOFAGENERALENGINE_SRC}/MergePoints.inl
    ${SOFAGENERALENGINE_SRC}/MergeROIs.h
    ${SOFAGENERALENGINE_SRC}/MergeSets.h
    ${SOFAGENERALENGINE_SRC}/MergeSets.inl
    ${SOFAGENERALENGINE_SRC}/MergeVectors.h
    ${SOFAGENERALENGINE_SRC}/MergeVectors.inl
    ${SOFAGENERALENGINE_SRC}/MeshBarycentricMapperEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshBarycentricMapperEngine.inl
    ${SOFAGENERALENGINE_SRC}/MeshClosingEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshClosingEngine.inl
    ${SOFAGENERALENGINE_SRC}/MeshBoundaryROI.h
    ${SOFAGENERALENGINE_SRC}/MeshROI.h
    ${SOFAGENERALENGINE_SRC}/MeshROI.inl
    ${SOFAGENERALENGINE_SRC}/MeshSampler.h
    ${SOFAGENERALENGINE_SRC}/MeshSampler.inl
    ${SOFAGENERALENGINE_SRC}/MeshSplittingEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshSplittingEngine.inl
    ${SOFAGENERALENGINE_SRC}/MeshSubsetEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshSubsetEngine.inl
    ${SOFAGENERALENGINE_SRC}/NearestPointROI.h
    ${SOFAGENERALENGINE_SRC}/NearestPointROI.inl
    ${SOFAGENERALENGINE_SRC}/NormEngine.h
    ${SOFAGENERALENGINE_SRC}/NormEngine.inl
    ${SOFAGENERALENGINE_SRC}/NormalsFromPoints.h
    ${SOFAGENERALENGINE_SRC}/NormalsFromPoints.inl
    ${SOFAGENERALENGINE_SRC}/PairBoxRoi.h
    ${SOFAGENERALENGINE_SRC}/PairBoxRoi.inl
    ${SOFAGENERALENGINE_SRC}/PlaneROI.h
    ${SOFAGENERALENGINE_SRC}/PlaneROI.inl
    ${SOFAGENERALENGINE_SRC}/PointsFromIndices.h
    ${SOFAGENERALENGINE_SRC}/PointsFromIndices.inl
    ${SOFAGENERALENGINE_SRC}/ProximityROI.h
    ${SOFAGENERALENGINE_SRC}/ProximityROI.inl
    ${SOFAGENERALENGINE_SRC}/QuatToRigidEngine.h
    ${SOFAGENERALENGINE_SRC}/QuatToRigidEngine.inl
    ${SOFAGENERALENGINE_SRC}/ROIValueMapper.h
    ${SOFAGENERALENGINE_SRC}/RandomPointDistributionInSurface.h
    ${SOFAGENERALENGINE_SRC}/RandomPointDistributionInSurface.inl
    ${SOFAGENERALENGINE_SRC}/RigidToQuatEngine.h
    ${SOFAGENERALENGINE_SRC}/RigidToQuatEngine.inl
    ${SOFAGENERALENGINE_SRC}/SelectLabelROI.h
    ${SOFAGENERALENGINE_SRC}/SelectConnectedLabelsROI.cpp
    ${SOFAGENERALENGINE_SRC}/ShapeMatching.h
    ${SOFAGENERALENGINE_SRC}/ShapeMatching.inl
    ${SOFAGENERALENGINE_SRC}/SmoothMeshEngine.h
    ${SOFAGENERALENGINE_SRC}/SmoothMeshEngine.inl
    ${SOFAGENERALENGINE_SRC}/SphereROI.h
    ${SOFAGENERALENGINE_SRC}/SphereROI.inl
    ${SOFAGENERALENGINE_SRC}/Spiral.h
    ${SOFAGENERALENGINE_SRC}/Spiral.inl
    ${SOFAGENERALENGINE_SRC}/SubsetTopology.h
    ${SOFAGENERALENGINE_SRC}/SubsetTopology.inl
    ${SOFAGENERALENGINE_SRC}/SumEngine.h
    ${SOFAGENERALENGINE_SRC}/SumEngine.inl
    ${SOFAGENERALENGINE_SRC}/TextureInterpolation.h
    ${SOFAGENERALENGINE_SRC}/TextureInterpolation.inl
    ${SOFAGENERALENGINE_SRC}/TransformEngine.h
    ${SOFAGENERALENGINE_SRC}/TransformEngine.inl
    ${SOFAGENERALENGINE_SRC}/TransformMatrixEngine.h
    ${SOFAGENERALENGINE_SRC}/TransformPosition.h
    ${SOFAGENERALENGINE_SRC}/TransformPosition.inl
    ${SOFAGENERALENGINE_SRC}/ValuesFromIndices.h
    ${SOFAGENERALENGINE_SRC}/ValuesFromIndices.inl
    ${SOFAGENERALENGINE_SRC}/ValuesFromPositions.h
    ${SOFAGENERALENGINE_SRC}/ValuesFromPositions.inl
    ${SOFAGENERALENGINE_SRC}/Vertex2Frame.h
    ${SOFAGENERALENGINE_SRC}/Vertex2Frame.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALENGINE_SRC}/AverageCoord.cpp
    ${SOFAGENERALENGINE_SRC}/ClusteringEngine.cpp
    ${SOFAGENERALENGINE_SRC}/ComplementaryROI.cpp
    ${SOFAGENERALENGINE_SRC}/DilateEngine.cpp
    ${SOFAGENERALENGINE_SRC}/DifferenceEngine.cpp
    ${SOFAGENERALENGINE_SRC}/ExtrudeEdgesAndGenerateQuads.cpp
    ${SOFAGENERALENGINE_SRC}/ExtrudeQuadsAndGenerateHexas.cpp
    ${SOFAGENERALENGINE_SRC}/ExtrudeSurface.cpp
    ${SOFAGENERALENGINE_SRC}/GenerateCylinder.cpp
    ${SOFAGENERALENGINE_SRC}/GenerateGrid.cpp
    ${SOFAGENERALENGINE_SRC}/GenerateRigidMass.cpp
    ${SOFAGENERALENGINE_SRC}/GenerateSphere.cpp
    ${SOFAGENERALENGINE_SRC}/GroupFilterYoungModulus.cpp
    ${SOFAGENERALENGINE_SRC}/HausdorffDistance.cpp
    ${SOFAGENERALENGINE_SRC}/IndexValueMapper.cpp
    ${SOFAGENERALENGINE_SRC}/Indices2ValuesMapper.cpp
    ${SOFAGENERALENGINE_SRC}/IndicesFromValues.cpp
    ${SOFAGENERALENGINE_SRC}/JoinPoints.cpp
    ${SOFAGENERALENGINE_SRC}/MapIndices.cpp
    ${SOFAGENERALENGINE_SRC}/MathOp.cpp
    ${SOFAGENERALENGINE_SRC}/MergeMeshes.cpp
    ${SOFAGENERALENGINE_SRC}/MergePoints.cpp
    ${SOFAGENERALENGINE_SRC}/MergeROIs.cpp
    ${SOFAGENERALENGINE_SRC}/MergeSets.cpp
    ${SOFAGENERALENGINE_SRC}/MergeVectors.cpp
    ${SOFAGENERALENGINE_SRC}/MeshBarycentricMapperEngine.cpp
    ${SOFAGENERALENGINE_SRC}/MeshClosingEngine.cpp
    ${SOFAGENERALENGINE_SRC}/MeshBoundaryROI.cpp
    ${SOFAGENERALENGINE_SRC}/MeshROI.cpp
    ${SOFAGENERALENGINE_SRC}/MeshSampler.cpp
    ${SOFAGENERALENGINE_SRC}/MeshSplittingEngine.cpp
    ${SOFAGENERALENGINE_SRC}/MeshSubsetEngine.cpp
    ${SOFAGENERALENGINE_SRC}/NearestPointROI.cpp
    ${SOFAGENERALENGINE_SRC}/NormEngine.cpp
    ${SOFAGENERALENGINE_SRC}/NormalsFromPoints.cpp
    ${SOFAGENERALENGINE_SRC}/PairBoxRoi.cpp
    ${SOFAGENERALENGINE_SRC}/PlaneROI.cpp
    ${SOFAGENERALENGINE_SRC}/PointsFromIndices.cpp
    ${SOFAGENERALENGINE_SRC}/ProximityROI.cpp
    ${SOFAGENERALENGINE_SRC}/QuatToRigidEngine.cpp
    ${SOFAGENERALENGINE_SRC}/ROIValueMapper.cpp
    ${SOFAGENERALENGINE_SRC}/RandomPointDistributionInSurface.cpp
    ${SOFAGENERALENGINE_SRC}/RigidToQuatEngine.cpp
    ${SOFAGENERALENGINE_SRC}/SelectLabelROI.cpp
    ${SOFAGENERALENGINE_SRC}/SelectConnectedLabelsROI.cpp
    ${SOFAGENERALENGINE_SRC}/ShapeMatching.cpp
    ${SOFAGENERALENGINE_SRC}/SmoothMeshEngine.cpp
    ${SOFAGENERALENGINE_SRC}/SphereROI.cpp
    ${SOFAGENERALENGINE_SRC}/Spiral.cpp
    ${SOFAGENERALENGINE_SRC}/SubsetTopology.cpp
    ${SOFAGENERALENGINE_SRC}/SumEngine.cpp
    ${SOFAGENERALENGINE_SRC}/TextureInterpolation.cpp
    ${SOFAGENERALENGINE_SRC}/TransformEngine.cpp
    ${SOFAGENERALENGINE_SRC}/TransformMatrixEngine.cpp
    ${SOFAGENERALENGINE_SRC}/TransformPosition.cpp
    ${SOFAGENERALENGINE_SRC}/ValuesFromIndices.cpp
    ${SOFAGENERALENGINE_SRC}/ValuesFromPositions.cpp
    ${SOFAGENERALENGINE_SRC}/Vertex2Frame.cpp
    )

find_package(SofaCommon REQUIRED) # SofaMeshCollision
find_package(SofaGeneralMeshCollision REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaGeneralMeshCollision)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAGENERALENGINE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGENERALENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()

