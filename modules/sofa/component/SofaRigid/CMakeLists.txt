include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initRigid.h 
    # ../container/ArticulatedHierarchyContainer.h 
    # ../container/ArticulatedHierarchyContainer.inl 
    # ../mapping/ArticulatedSystemMapping.h 
    # ../mapping/ArticulatedSystemMapping.inl 
    # ../mapping/LaparoscopicRigidMapping.h 
    # ../mapping/LaparoscopicRigidMapping.inl 
    # ../mapping/LineSetSkinningMapping.h 
    # ../mapping/LineSetSkinningMapping.inl 
    ../mapping/RigidMapping.h 
    ../mapping/RigidMapping.inl 
    # ../mapping/RigidRigidMapping.h 
    # ../mapping/RigidRigidMapping.inl 
    # ../mapping/SkinningMapping.h 
    # ../mapping/SkinningMapping.inl 
     ../interactionforcefield/JointSpringForceField.h
     ../interactionforcefield/JointSpringForceField.inl

    )
    
set(SOURCE_FILES

    ../initRigid.cpp 
    # ../container/ArticulatedHierarchyContainer.cpp 
    # ../mapping/ArticulatedSystemMapping.cpp 
    # ../mapping/LaparoscopicRigidMapping.cpp 
    # ../mapping/LineSetSkinningMapping.cpp 
    ../mapping/RigidMapping.cpp 
    # ../mapping/RigidRigidMapping.cpp 
    # ../mapping/SkinningMapping.cpp 
     ../interactionforcefield/JointSpringForceField.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_RIGID")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaBaseMechanics)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
