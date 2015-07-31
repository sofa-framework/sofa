include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initRigid.h
    ArticulatedHierarchyContainer.h
    ArticulatedHierarchyContainer.inl
    ArticulatedSystemMapping.h
    ArticulatedSystemMapping.inl
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
    JointSpringForceField.h
    JointSpringForceField.inl

    )

set(SOURCE_FILES

    initRigid.cpp
    ArticulatedHierarchyContainer.cpp
    ArticulatedSystemMapping.cpp
    LaparoscopicRigidMapping.cpp
    LineSetSkinningMapping.cpp
    RigidMapping.cpp
    RigidRigidMapping.cpp
    SkinningMapping.cpp
    JointSpringForceField.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_RIGID")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaBaseMechanics)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
