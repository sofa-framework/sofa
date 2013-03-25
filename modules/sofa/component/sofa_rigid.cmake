cmake_minimum_required(VERSION 2.8)

project("SofaRigid")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initRigid.h 
    container/ArticulatedHierarchyContainer.h 
    container/ArticulatedHierarchyContainer.inl 
    mapping/ArticulatedSystemMapping.h 
    mapping/ArticulatedSystemMapping.inl 
    mapping/LaparoscopicRigidMapping.h 
    mapping/LaparoscopicRigidMapping.inl 
    mapping/LineSetSkinningMapping.h 
    mapping/LineSetSkinningMapping.inl 
    mapping/RigidMapping.h 
    mapping/RigidMapping.inl 
    mapping/RigidRigidMapping.h 
    mapping/RigidRigidMapping.inl 
    mapping/SkinningMapping.h 
    mapping/SkinningMapping.inl 
    interactionforcefield/JointSpringForceField.h 
    interactionforcefield/JointSpringForceField.inl

    )
    
set(SOURCE_FILES

    initRigid.cpp 
    container/ArticulatedHierarchyContainer.cpp 
    mapping/ArticulatedSystemMapping.cpp 
    mapping/LaparoscopicRigidMapping.cpp 
    mapping/LineSetSkinningMapping.cpp 
    mapping/RigidMapping.cpp 
    mapping/RigidRigidMapping.cpp 
    mapping/SkinningMapping.cpp 
    interactionforcefield/JointSpringForceField.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaEigen2Solver )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_RIGID")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
