cmake_minimum_required(VERSION 3.12)
project(SofaObjectInteraction LANGUAGES CXX)

set(SOFAOBJECTINTERACTION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAOBJECTINTERACTION_SRC}/config.h.in
    ${SOFAOBJECTINTERACTION_SRC}/initSofaObjectInteraction.h
    ${SOFAOBJECTINTERACTION_SRC}/PenalityContactForceField.h
    ${SOFAOBJECTINTERACTION_SRC}/PenalityContactForceField.inl
)

set(SOURCE_FILES
    ${SOFAOBJECTINTERACTION_SRC}/initSofaObjectInteraction.cpp
    ${SOFAOBJECTINTERACTION_SRC}/PenalityContactForceField.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaDeformable)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
