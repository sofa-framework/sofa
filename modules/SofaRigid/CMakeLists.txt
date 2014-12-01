cmake_minimum_required(VERSION 2.8)
project(SofaRigid)

set(HEADER_FILES

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

target_link_libraries( ${PROJECT_NAME} SofaBaseMechanics)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFARIGID_COMPILER_FLAGS "-DSOFA_BUILD_RIGID")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFARIGID_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}" COMPONENT SofaCommon_libraries
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        PUBLIC_HEADER DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaCommon_headers
        )
