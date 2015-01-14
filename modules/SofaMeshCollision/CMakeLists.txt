cmake_minimum_required(VERSION 2.8.12)
project(SofaMeshCollision)

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

target_link_libraries( ${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAMESHCOLLISION_COMPILER_FLAGS "-DSOFA_BUILD_MESH_COLLISION")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAMESHCOLLISION_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}" COMPONENT SofaCommon_libraries
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        PUBLIC_HEADER DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaCommon_headers
        )