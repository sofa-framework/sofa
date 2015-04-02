include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initMeshCollision.h
    MeshNewProximityIntersection.h
    MeshNewProximityIntersection.inl
    BarycentricPenalityContact.h
    BarycentricPenalityContact.inl
    BarycentricContactMapper.h
    BarycentricContactMapper.inl
    IdentityContactMapper.h
    IdentityContactMapper.inl
    RigidContactMapper.h
    RigidContactMapper.inl
    SubsetContactMapper.h
    SubsetContactMapper.inl
    MeshDiscreteIntersection.h
    MeshDiscreteIntersection.inl
    MeshMinProximityIntersection.h
    Point.h
    PointModel.h
    PointModel.inl
    Line.h
    LineModel.h
    LineModel.inl
    Triangle.h
    TriangleModel.h
    TriangleModel.inl
    TriangleOctree.h
    TriangleOctreeModel.h
    RayTriangleIntersection.h
    PointLocalMinDistanceFilter.h
    LineLocalMinDistanceFilter.h
    TriangleLocalMinDistanceFilter.h
    LocalMinDistanceFilter.h
    MeshIntTool.h
    MeshIntTool.inl
    IntrTriangleOBB.h
    IntrTriangleOBB.inl
    IntrMeshUtility.h
    IntrMeshUtility.inl
    IncrSAP.h
    IncrSAP.inl
    DirectSAP.h
    DirectSAP.inl
    )

set(SOURCE_FILES

    initMeshCollision.cpp
    MeshNewProximityIntersection.cpp
    BarycentricPenalityContact.cpp
    BarycentricContactMapper.cpp
    IdentityContactMapper.cpp
    SubsetContactMapper.cpp
    MeshDiscreteIntersection.cpp
    MeshMinProximityIntersection.cpp
    PointModel.cpp
    LineModel.cpp
    TriangleModel.cpp
    TriangleOctree.cpp
    TriangleOctreeModel.cpp
    RayTriangleIntersection.cpp
    PointLocalMinDistanceFilter.cpp
    LineLocalMinDistanceFilter.cpp
    TriangleLocalMinDistanceFilter.cpp
    LocalMinDistanceFilter.cpp
    MeshIntTool.cpp
    RigidContactMapper.cpp
    IntrTriangleOBB.cpp
    IntrMeshUtility.cpp
    IncrSAP.cpp
    DirectSAP.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MESH_COLLISION")
AddLinkerDependencies(SofaObjectInteraction SofaBaseCollision)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
