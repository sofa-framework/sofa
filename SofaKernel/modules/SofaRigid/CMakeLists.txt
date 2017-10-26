cmake_minimum_required(VERSION 3.1)
project(SofaRigid)

set(HEADER_FILES
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
    JointSpringForceField.cpp
    RigidMapping.cpp
    RigidRigidMapping.cpp
    initRigid.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_RIGID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
