cmake_minimum_required(VERSION 3.12)
project(SofaMeshCollision LANGUAGES CXX)

set(SOFAMESHCOLLISION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMESHCOLLISION_SRC}/config.h.in
    ${SOFAMESHCOLLISION_SRC}/initSofaMeshCollision.h
    ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.inl
    ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.h
    ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.inl
    ${SOFAMESHCOLLISION_SRC}/CollisionPM.h
    ${SOFAMESHCOLLISION_SRC}/EndPoint.h
    ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.inl
    ${SOFAMESHCOLLISION_SRC}/IntrMeshUtility.h
    ${SOFAMESHCOLLISION_SRC}/IntrMeshUtility.inl
    ${SOFAMESHCOLLISION_SRC}/IntrTriangleOBB.h
    ${SOFAMESHCOLLISION_SRC}/IntrTriangleOBB.inl
    ${SOFAMESHCOLLISION_SRC}/Line.h
    ${SOFAMESHCOLLISION_SRC}/LineLocalMinDistanceFilter.h
    ${SOFAMESHCOLLISION_SRC}/LineModel.h
    ${SOFAMESHCOLLISION_SRC}/LineModel.inl
    ${SOFAMESHCOLLISION_SRC}/LocalMinDistanceFilter.h
    ${SOFAMESHCOLLISION_SRC}/MeshIntTool.h
    ${SOFAMESHCOLLISION_SRC}/MeshIntTool.inl
    ${SOFAMESHCOLLISION_SRC}/MeshNewProximityIntersection.h
    ${SOFAMESHCOLLISION_SRC}/MeshNewProximityIntersection.inl
    ${SOFAMESHCOLLISION_SRC}/Point.h
    ${SOFAMESHCOLLISION_SRC}/PointLocalMinDistanceFilter.h
    ${SOFAMESHCOLLISION_SRC}/PointModel.h
    ${SOFAMESHCOLLISION_SRC}/PointModel.inl
    ${SOFAMESHCOLLISION_SRC}/RayTriangleIntersection.h
    ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.inl
    ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.h
    ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.inl
    ${SOFAMESHCOLLISION_SRC}/Triangle.h
    ${SOFAMESHCOLLISION_SRC}/TriangleLocalMinDistanceFilter.h
    ${SOFAMESHCOLLISION_SRC}/TriangleModel.h
    ${SOFAMESHCOLLISION_SRC}/TriangleModel.inl
)

set(SOURCE_FILES
    ${SOFAMESHCOLLISION_SRC}/initSofaMeshCollision.cpp
    ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.cpp
    ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.cpp
    ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.cpp
    ${SOFAMESHCOLLISION_SRC}/IntrMeshUtility.cpp
    ${SOFAMESHCOLLISION_SRC}/IntrTriangleOBB.cpp
    ${SOFAMESHCOLLISION_SRC}/LineLocalMinDistanceFilter.cpp
    ${SOFAMESHCOLLISION_SRC}/LineModel.cpp
    ${SOFAMESHCOLLISION_SRC}/LocalMinDistanceFilter.cpp
    ${SOFAMESHCOLLISION_SRC}/MeshIntTool.cpp
    ${SOFAMESHCOLLISION_SRC}/MeshNewProximityIntersection.cpp
    ${SOFAMESHCOLLISION_SRC}/PointLocalMinDistanceFilter.cpp
    ${SOFAMESHCOLLISION_SRC}/PointModel.cpp
    ${SOFAMESHCOLLISION_SRC}/RayTriangleIntersection.cpp
    ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.cpp
    ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.cpp
    ${SOFAMESHCOLLISION_SRC}/TriangleLocalMinDistanceFilter.cpp
    ${SOFAMESHCOLLISION_SRC}/TriangleModel.cpp
)

find_package(SofaBase REQUIRED) # SofaBaseMechanics SofaBaseCollision SofaBaseTopology
find_package(SofaObjectInteraction REQUIRED)
find_package(SofaRigid REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision SofaBaseMechanics SofaBaseTopology)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMESHCOLLISION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMESHCOLLISION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
