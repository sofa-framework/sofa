cmake_minimum_required(VERSION 3.1)
project(SofaGeneralRigid)

set(HEADER_FILES
    ArticulatedHierarchyContainer.h
    ArticulatedHierarchyContainer.inl
    ArticulatedSystemMapping.h
    ArticulatedSystemMapping.inl
    LaparoscopicRigidMapping.h
    LaparoscopicRigidMapping.inl
    LineSetSkinningMapping.h
    LineSetSkinningMapping.inl
    SkinningMapping.h
    SkinningMapping.inl
    config.h
    initGeneralRigid.h
)

set(SOURCE_FILES
    ArticulatedHierarchyContainer.cpp
    ArticulatedSystemMapping.cpp
    LaparoscopicRigidMapping.cpp
    LineSetSkinningMapping.cpp
    SkinningMapping.cpp
    initGeneralRigid.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_RIGID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
