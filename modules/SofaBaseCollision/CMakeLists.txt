cmake_minimum_required(VERSION 2.8.11)
project(SofaBaseCollision)

set(HEADER_FILES

    initBaseCollision.h
    BaseContactMapper.h
    DefaultPipeline.h
    Sphere.h
    SphereModel.h
    SphereModel.inl
    Cube.h
    CubeModel.h
    DiscreteIntersection.h
    BruteForceDetection.h
    DefaultContactManager.h
    MinProximityIntersection.h
    NewProximityIntersection.h
    NewProximityIntersection.inl
    CapsuleModel.h
    CapsuleModel.inl
    BaseProximityIntersection.h
    CapsuleIntTool.h
    CapsuleIntTool.inl
    OBBModel.inl
    OBBModel.h
	RigidCapsuleModel.h
	RigidCapsuleModel.inl
	CylinderModel.h
	CylinderModel.inl
    OBBIntTool.h
    IntrOBBOBB.h
    IntrOBBOBB.inl
    IntrUtility3.h
    IntrUtility3.inl
    IntrSphereOBB.h
    IntrCapsuleOBB.h
    IntrCapsuleOBB.inl
    IntrSphereOBB.inl
    Intersector.h
    BaseIntTool.h
    ContactListener.h
    )

set(SOURCE_FILES

    initBaseCollision.cpp
    BaseContactMapper.cpp
    DefaultPipeline.cpp
    SphereModel.cpp
    CubeModel.cpp
    CapsuleModel.cpp
    DiscreteIntersection.cpp
    BruteForceDetection.cpp
    DefaultContactManager.cpp
    MinProximityIntersection.cpp
    NewProximityIntersection.cpp
    BaseProximityIntersection.cpp
    CapsuleIntTool.cpp
    OBBModel.cpp
	RigidCapsuleModel.cpp
	CylinderModel.cpp
    OBBIntTool.cpp
    IntrOBBOBB.cpp
    IntrUtility3.cpp
    IntrCapsuleOBB.cpp
    IntrSphereOBB.cpp
    BaseIntTool.cpp
    ContactListener.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaSimulationCommon)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFABASE_DEPENDENCY_INCLUDE_DIR}>")

set( COMPILER_FLAGS "-DSOFA_BUILD_BASE_COLLISION")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFABASE_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaBaseTargets
        RUNTIME DESTINATION "${SOFABASE_INSTALL_BIN_DIR}" COMPONENT SofaBase_libraries
        LIBRARY DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        ARCHIVE DESTINATION "${SOFABASE_INSTALL_LIB_DIR}" COMPONENT SofaBase_libraries
        PUBLIC_HEADER DESTINATION "${SOFABASE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaBase_headers
        )
