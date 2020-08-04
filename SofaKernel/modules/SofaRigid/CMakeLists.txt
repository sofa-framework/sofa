cmake_minimum_required(VERSION 3.12)
project(SofaRigid)

set(HEADER_FILES
    JointSpring.h
    JointSpring.inl
    JointSpringForceField.h
    JointSpringForceField.inl
    RigidMapping.h
    RigidMapping.inl
    RigidRigidMapping.h
    RigidRigidMapping.inl
    config.h
    initRigid.h
)

set(SOURCE_FILES
    JointSpring.cpp
    JointSpringForceField.cpp
    RigidMapping.cpp
    RigidRigidMapping.cpp
    initRigid.cpp
)

find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
