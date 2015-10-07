cmake_minimum_required(VERSION 2.8.12)
project(SofaRigid)

set(HEADER_FILES
    ArticulatedHierarchyContainer.h
    ArticulatedHierarchyContainer.inl
    ArticulatedSystemMapping.h
    ArticulatedSystemMapping.inl
    JointSpringForceField.h
    JointSpringForceField.inl
    LaparoscopicRigidMapping.h
    LaparoscopicRigidMapping.inl
    LineSetSkinningMapping.h
    LineSetSkinningMapping.inl
    RigidMapping.h
    RigidMapping.inl
    RigidRigidMapping.h
    RigidRigidMapping.inl
    SkinningMapping.h
    SkinningMapping.inl
    config.h
    initRigid.h
)

set(SOURCE_FILES
    ArticulatedHierarchyContainer.cpp
    ArticulatedSystemMapping.cpp
    JointSpringForceField.cpp
    LaparoscopicRigidMapping.cpp
    LineSetSkinningMapping.cpp
    RigidMapping.cpp
    RigidRigidMapping.cpp
    SkinningMapping.cpp
    initRigid.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_RIGID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
