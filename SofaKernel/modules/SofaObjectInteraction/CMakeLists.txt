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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OBJECT_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
