cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMeshCollision.h 
    ../collision/MeshNewProximityIntersection.h 
    ../collision/MeshNewProximityIntersection.inl 
    ../collision/BarycentricPenalityContact.h 
    ../collision/BarycentricPenalityContact.inl 
    ../collision/BarycentricContactMapper.h 
    ../collision/BarycentricContactMapper.inl 
    ../collision/IdentityContactMapper.h 
    ../collision/IdentityContactMapper.inl 
    ../collision/RigidContactMapper.h 
    ../collision/RigidContactMapper.inl 
    ../collision/SubsetContactMapper.h 
    ../collision/SubsetContactMapper.inl 
    ../collision/MeshDiscreteIntersection.h 
    ../collision/MeshDiscreteIntersection.inl 
    ../collision/MeshMinProximityIntersection.h 
    ../collision/Point.h 
    ../collision/PointModel.h 
    ../collision/PointModel.inl 
    ../collision/Line.h 
    ../collision/LineModel.h 
    ../collision/LineModel.inl 
    ../collision/Triangle.h 
    ../collision/TriangleModel.h 
    ../collision/TriangleModel.inl 
    ../collision/TriangleOctree.h 
    ../collision/TriangleOctreeModel.h 
    ../collision/RayTriangleIntersection.h 
    ../collision/PointLocalMinDistanceFilter.h 
    ../collision/LineLocalMinDistanceFilter.h 
    ../collision/TriangleLocalMinDistanceFilter.h 
    ../collision/LocalMinDistanceFilter.h 
    ../collision/MeshIntTool.h 
    ../collision/IntrTriangleOBB.h 
    ../collision/IntrTriangleOBB.inl 
    ../collision/IntrMeshUtility.h 
    ../collision/IntrMeshUtility.inl

    )
    
set(SOURCE_FILES

    ../initMeshCollision.cpp 
    ../collision/MeshNewProximityIntersection.cpp 
    ../collision/BarycentricPenalityContact.cpp 
    ../collision/BarycentricContactMapper.cpp 
    ../collision/IdentityContactMapper.cpp 
    ../collision/SubsetContactMapper.cpp 
    ../collision/MeshDiscreteIntersection.cpp 
    ../collision/MeshMinProximityIntersection.cpp 
    ../collision/PointModel.cpp 
    ../collision/LineModel.cpp 
    ../collision/TriangleModel.cpp 
    ../collision/TriangleOctree.cpp 
    ../collision/TriangleOctreeModel.cpp 
    ../collision/RayTriangleIntersection.cpp 
    ../collision/PointLocalMinDistanceFilter.cpp 
    ../collision/LineLocalMinDistanceFilter.cpp 
    ../collision/TriangleLocalMinDistanceFilter.cpp 
    ../collision/LocalMinDistanceFilter.cpp 
    ../collision/MeshIntTool.cpp 
    ../collision/RigidContactMapper.cpp 
    ../collision/IntrTriangleOBB.cpp 
    ../collision/IntrMeshUtility.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_MESH_COLLISION")
set(LINKER_DEPENDENCIES SofaObjectInteraction SofaBaseCollision)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
