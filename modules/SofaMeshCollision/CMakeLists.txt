cmake_minimum_required(VERSION 2.8.12)
project(SofaMeshCollision)

set(HEADER_FILES
    BarycentricContactMapper.h
    BarycentricContactMapper.inl
    BarycentricPenalityContact.h
    BarycentricPenalityContact.inl
    CollisionPM.h
    DirectSAP.h
    DirectSAP.inl
    EndPoint.h
    IdentityContactMapper.h
    IdentityContactMapper.inl
    IncrSAP.h
    IncrSAP.inl
    IntrMeshUtility.h
    IntrMeshUtility.inl
    IntrTriangleOBB.h
    IntrTriangleOBB.inl
    Line.h
    LineLocalMinDistanceFilter.h
    LineModel.h
    LineModel.inl
    LocalMinDistanceFilter.h
    MeshDiscreteIntersection.h
    MeshDiscreteIntersection.inl
    MeshIntTool.h
    MeshIntTool.inl
    MeshMinProximityIntersection.h
    MeshNewProximityIntersection.h
    MeshNewProximityIntersection.inl
    Point.h
    PointLocalMinDistanceFilter.h
    PointModel.h
    PointModel.inl
    RayTriangleIntersection.h
    RigidContactMapper.h
    RigidContactMapper.inl
    SubsetContactMapper.h
    SubsetContactMapper.inl
    Triangle.h
    TriangleLocalMinDistanceFilter.h
    TriangleModel.h
    TriangleModel.inl
    TriangleOctree.h
    TriangleOctreeModel.h
    config.h
    initMeshCollision.h
)

set(SOURCE_FILES
    BarycentricContactMapper.cpp
    BarycentricPenalityContact.cpp
    DirectSAP.cpp
    IdentityContactMapper.cpp
    IncrSAP.cpp
    IntrMeshUtility.cpp
    IntrTriangleOBB.cpp
    LineLocalMinDistanceFilter.cpp
    LineModel.cpp
    LocalMinDistanceFilter.cpp
    MeshDiscreteIntersection.cpp
    MeshIntTool.cpp
    MeshMinProximityIntersection.cpp
    MeshNewProximityIntersection.cpp
    PointLocalMinDistanceFilter.cpp
    PointModel.cpp
    RayTriangleIntersection.cpp
    RigidContactMapper.cpp
    SubsetContactMapper.cpp
    TriangleLocalMinDistanceFilter.cpp
    TriangleModel.cpp
    TriangleOctree.cpp
    TriangleOctreeModel.cpp
    initMeshCollision.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MESH_COLLISION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaCommon_install(${PROJECT_NAME})
