cmake_minimum_required(VERSION 3.12)
project(SofaGeneralMeshCollision)

set(HEADER_FILES
    config.h
    initGeneralMeshCollision.h
    )

set(SOURCE_FILES
    initGeneralMeshCollision.cpp
    )

list(APPEND HEADER_FILES
    DirectSAP.h
    IncrSAP.h
    # IntrTriangleOBB.h
    # IntrTriangleOBB.inl
    MeshDiscreteIntersection.h
    MeshDiscreteIntersection.inl
    MeshMinProximityIntersection.h
    # RigidContactMapper.h
    # RigidContactMapper.inl
    # SubsetContactMapper.h
    # SubsetContactMapper.inl
    TriangleOctree.h
    TriangleOctreeModel.h
    )
list(APPEND SOURCE_FILES
    DirectSAP.cpp
    IncrSAP.cpp
    # IntrTriangleOBB.cpp
    MeshDiscreteIntersection.cpp
    MeshMinProximityIntersection.cpp
    # RigidContactMapper.cpp
    # SubsetContactMapper.cpp
    TriangleOctree.cpp
    TriangleOctreeModel.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision SofaMeshCollision)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGeneral
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
