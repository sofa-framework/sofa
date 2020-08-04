cmake_minimum_required(VERSION 3.12)
project(SofaObjectInteraction)

set(HEADER_FILES
    PenalityContactForceField.h
    PenalityContactForceField.inl
    config.h
    initObjectInteraction.h
)

set(SOURCE_FILES
    PenalityContactForceField.cpp
    initObjectInteraction.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaDeformable)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
