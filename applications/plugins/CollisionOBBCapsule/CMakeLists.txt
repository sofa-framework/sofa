cmake_minimum_required(VERSION 3.22)
project(CollisionOBBCapsule VERSION 1.0 LANGUAGES CXX)

# List all files
set(COLLISIONOBBCAPSULE_SRC_DIR src/CollisionOBBCapsule)
set(HEADER_FILES
    ${COLLISIONOBBCAPSULE_SRC_DIR}/config.h.in
    ${COLLISIONOBBCAPSULE_SRC_DIR}/init.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/BaseIntTool.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/CapsuleIntTool.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/CapsuleIntTool.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/CapsuleIntersection.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/Intersector.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrCapsuleOBB.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrCapsuleOBB.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrMeshUtility.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrMeshUtility.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrOBBOBB.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrOBBOBB.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrSphereOBB.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrSphereOBB.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrTriangleOBB.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrTriangleOBB.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrUtility3.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrUtility3.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/MeshIntTool.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/MeshIntTool.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/OBBIntTool.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/OBBIntersection.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/CapsuleModel.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/CapsuleModel.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/OBBModel.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/OBBModel.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/RigidCapsuleModel.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/RigidCapsuleModel.inl
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/mapper/CapsuleContactMapper.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/mapper/OBBContactMapper.h
    )

set(SOURCE_FILES
    ${COLLISIONOBBCAPSULE_SRC_DIR}/init.cpp        
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/BaseIntTool.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/CapsuleIntTool.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/CapsuleIntersection.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/CapsuleIntersection.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrCapsuleOBB.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrMeshUtility.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrOBBOBB.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrSphereOBB.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrTriangleOBB.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/IntrUtility3.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/MeshIntTool.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/OBBIntTool.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/detection/intersection/OBBIntersection.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/CapsuleModel.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/OBBModel.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/geometry/RigidCapsuleModel.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/contact/CapsuleContact.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/contact/OBBContact.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/mapper/CapsuleContactMapper.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/mapper/CapsuleContactMapper.h
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/mapper/OBBContactMapper.cpp
    ${COLLISIONOBBCAPSULE_SRC_DIR}/response/mapper/OBBContactMapper.h
    )

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Component.Collision REQUIRED)
sofa_find_package(Sofa.GUI.Component REQUIRED) # for performers

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision Sofa.GUI.Component)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(COLLISIONOBBCAPSULE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(COLLISIONOBBCAPSULE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(CollisionOBBCapsule_test)
endif()
