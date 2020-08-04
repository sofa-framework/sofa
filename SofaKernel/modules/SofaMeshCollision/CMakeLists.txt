cmake_minimum_required(VERSION 3.12)
project(SofaMeshCollision)

set(HEADER_FILES
    BarycentricContactMapper.h
    BarycentricContactMapper.inl
    BarycentricPenalityContact.h
    BarycentricPenalityContact.inl
    CollisionPM.h
    EndPoint.h
    IdentityContactMapper.h
    IdentityContactMapper.inl
    IntrMeshUtility.h
    IntrMeshUtility.inl
    IntrTriangleOBB.h
    IntrTriangleOBB.inl
    Line.h
    LineLocalMinDistanceFilter.h
    LineModel.h
    LineModel.inl
    LocalMinDistanceFilter.h
    MeshIntTool.h
    MeshIntTool.inl
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
    config.h
    initMeshCollision.h
)

set(SOURCE_FILES
    BarycentricContactMapper.cpp
    BarycentricPenalityContact.cpp
    IdentityContactMapper.cpp
    IntrMeshUtility.cpp
    IntrTriangleOBB.cpp
    LineLocalMinDistanceFilter.cpp
    LineModel.cpp
    LocalMinDistanceFilter.cpp
    MeshIntTool.cpp
    MeshNewProximityIntersection.cpp
    PointLocalMinDistanceFilter.cpp
    PointModel.cpp
    RayTriangleIntersection.cpp
    RigidContactMapper.cpp
    SubsetContactMapper.cpp
    TriangleLocalMinDistanceFilter.cpp
    TriangleModel.cpp
    initMeshCollision.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
