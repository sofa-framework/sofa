cmake_minimum_required(VERSION 2.8.12)
project(SofaObjectInteraction)

set(HEADER_FILES

    initObjectInteraction.h
    AttachConstraint.h
    AttachConstraint.inl
    BoxStiffSpringForceField.h
    BoxStiffSpringForceField.inl
    InteractionEllipsoidForceField.h
    InteractionEllipsoidForceField.inl
    PenalityContactForceField.h
    PenalityContactForceField.inl
    RepulsiveSpringForceField.h
    RepulsiveSpringForceField.inl

    )

set(SOURCE_FILES

    initObjectInteraction.cpp
    AttachConstraint.cpp
    BoxStiffSpringForceField.cpp
    InteractionEllipsoidForceField.cpp
    PenalityContactForceField.cpp
    RepulsiveSpringForceField.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} SofaDeformable)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFACOMMON_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAOBJECTINTERACTION_COMPILER_FLAGS "-DSOFA_BUILD_OBJECT_INTERACTION")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFACOMMON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAOBJECTINTERACTION_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaCommonTargets
        RUNTIME DESTINATION "${SOFACOMMON_INSTALL_BIN_DIR}" COMPONENT SofaCommon_libraries
        LIBRARY DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        ARCHIVE DESTINATION "${SOFACOMMON_INSTALL_LIB_DIR}" COMPONENT SofaCommon_libraries
        PUBLIC_HEADER DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaCommon_headers
        )
