cmake_minimum_required(VERSION 3.1)
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

sofa_install_targets(SofaCommon ${PROJECT_NAME} "SofaCommon/${PROJECT_NAME}")
