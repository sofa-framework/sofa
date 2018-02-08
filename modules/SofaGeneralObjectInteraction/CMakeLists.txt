cmake_minimum_required(VERSION 3.1)
project(SofaGeneralObjectInteraction)

set(HEADER_FILES
    config.h
    initGeneralObjectInteraction.h
    )

set(SOURCE_FILES
    initGeneralObjectInteraction.cpp
    )

list(APPEND HEADER_FILES
    AttachConstraint.h
    AttachConstraint.inl
    BoxStiffSpringForceField.h
    BoxStiffSpringForceField.inl
    InteractionEllipsoidForceField.h
    InteractionEllipsoidForceField.inl
    RepulsiveSpringForceField.h
    RepulsiveSpringForceField.inl
    )
list(APPEND SOURCE_FILES
    AttachConstraint.cpp
    BoxStiffSpringForceField.cpp
    InteractionEllipsoidForceField.cpp
    RepulsiveSpringForceField.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaDeformable)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_OBJECT_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
