cmake_minimum_required(VERSION 3.12)
project(SofaGeneralObjectInteraction LANGUAGES CXX)

set(SOFAGENERALOBJECTINTERACTION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALOBJECTINTERACTION_SRC}/config.h.in
    ${SOFAGENERALOBJECTINTERACTION_SRC}/initSofaGeneralObjectInteraction.h
    )

set(SOURCE_FILES
    ${SOFAGENERALOBJECTINTERACTION_SRC}/initSofaGeneralObjectInteraction.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALOBJECTINTERACTION_SRC}/AttachConstraint.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/AttachConstraint.inl
    ${SOFAGENERALOBJECTINTERACTION_SRC}/BoxStiffSpringForceField.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/BoxStiffSpringForceField.inl
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.inl
    ${SOFAGENERALOBJECTINTERACTION_SRC}/RepulsiveSpringForceField.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/RepulsiveSpringForceField.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALOBJECTINTERACTION_SRC}/AttachConstraint.cpp
    ${SOFAGENERALOBJECTINTERACTION_SRC}/BoxStiffSpringForceField.cpp
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.cpp
    ${SOFAGENERALOBJECTINTERACTION_SRC}/RepulsiveSpringForceField.cpp
    )

find_package(SofaCommon REQUIRED) # SofaDeformable

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
