cmake_minimum_required(VERSION 2.8.12)
project(SofaObjectInteraction)

set(HEADER_FILES
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
    config.h
    initObjectInteraction.h
)

set(SOURCE_FILES
    AttachConstraint.cpp
    BoxStiffSpringForceField.cpp
    InteractionEllipsoidForceField.cpp
    PenalityContactForceField.cpp
    RepulsiveSpringForceField.cpp
    initObjectInteraction.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaDeformable)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OBJECT_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
