cmake_minimum_required(VERSION 2.8)
project(SofaEngine)

set(HEADER_FILES

    initEngine.h
    AverageCoord.h
    AverageCoord.inl
    BoxROI.h
    BoxROI.inl
    PairBoxRoi.h
    PairBoxRoi.inl
    PlaneROI.h
    PlaneROI.inl
    SphereROI.h
    SphereROI.inl
    DilateEngine.h
    DilateEngine.inl
    GenerateCylinder.h
    GenerateCylinder.inl
    ExtrudeSurface.h
    ExtrudeSurface.inl
    ExtrudeQuadsAndGenerateHexas.h
    ExtrudeQuadsAndGenerateHexas.inl
    ExtrudeEdgesAndGenerateQuads.h
    ExtrudeEdgesAndGenerateQuads.inl
    GenerateRigidMass.h
    GenerateRigidMass.inl
    GroupFilterYoungModulus.h
    GroupFilterYoungModulus.inl
    MathOp.h
    MathOp.inl
    MergeMeshes.h
    MergeMeshes.inl
    MergePoints.h
    MergePoints.inl
    MergeSets.h
    MergeSets.inl
    MergeVectors.h
    MergeVectors.inl
    MeshBarycentricMapperEngine.h
    MeshBarycentricMapperEngine.inl
    MeshROI.h
    MeshROI.inl
    TransformPosition.h
    TransformPosition.inl
    TransformEngine.h
    TransformEngine.inl
    TransformMatrixEngine.h
    PointsFromIndices.h
    PointsFromIndices.inl
    ValuesFromIndices.h
    ValuesFromIndices.inl
    IndicesFromValues.h
    IndicesFromValues.inl
    IndexValueMapper.h
    IndexValueMapper.inl
    JoinPoints.h
    JoinPoints.inl
    MapIndices.h
    MapIndices.inl
    RandomPointDistributionInSurface.h
    RandomPointDistributionInSurface.inl
    SmoothMeshEngine.h
    SmoothMeshEngine.inl
    Spiral.h
    Spiral.inl
    Vertex2Frame.h
    Vertex2Frame.inl
    TextureInterpolation.h
    TextureInterpolation.inl
    SubsetTopology.h
    SubsetTopology.inl
    RigidToQuatEngine.h
    RigidToQuatEngine.inl
    QuatToRigidEngine.h
    QuatToRigidEngine.inl
    ValuesFromPositions.h
    ValuesFromPositions.inl
    NormalsFromPoints.h
    NormalsFromPoints.inl
    ClusteringEngine.h
    ClusteringEngine.inl
    ShapeMatching.h
    ShapeMatching.inl
    ProximityROI.h
    ProximityROI.inl
    HausdorffDistance.h
    HausdorffDistance.inl
    NormEngine.h
    NormEngine.inl
    TestEngine.h
    )

set(SOURCE_FILES

    initEngine.cpp
    AverageCoord.cpp
    BoxROI.cpp
    PairBoxRoi.cpp
    PlaneROI.cpp
    SphereROI.cpp
    DilateEngine.cpp
    GenerateCylinder.cpp
    ExtrudeSurface.cpp
    ExtrudeQuadsAndGenerateHexas.cpp
    ExtrudeEdgesAndGenerateQuads.cpp
    GenerateRigidMass.cpp
    GroupFilterYoungModulus.cpp
    MathOp.cpp
    MergeMeshes.cpp
    MergePoints.cpp
    MergeSets.cpp
    MergeVectors.cpp
    MeshBarycentricMapperEngine.cpp
    MeshROI.cpp
    TransformPosition.cpp
    TransformEngine.cpp
    TransformMatrixEngine.cpp
    PointsFromIndices.cpp
    ValuesFromIndices.cpp
    IndicesFromValues.cpp
    IndexValueMapper.cpp
    JoinPoints.cpp
    MapIndices.cpp
    RandomPointDistributionInSurface.cpp
    SmoothMeshEngine.cpp
    Spiral.cpp
    Vertex2Frame.cpp
    TextureInterpolation.cpp
    SubsetTopology.cpp
    RigidToQuatEngine.cpp
    QuatToRigidEngine.cpp
    ValuesFromPositions.cpp
    NormalsFromPoints.cpp
    ClusteringEngine.cpp
    ShapeMatching.cpp
    ProximityROI.cpp
    HausdorffDistance.cpp
    NormEngine.cpp
    TestEngine.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaMeshCollision )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAENGINE_COMPILER_FLAGS "-DSOFA_BUILD_ENGINE")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAENGINE_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaEngine_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"

)
