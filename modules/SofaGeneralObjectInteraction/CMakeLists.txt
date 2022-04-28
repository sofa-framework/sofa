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
    ${SOFAGENERALOBJECTINTERACTION_SRC}/BoxStiffSpringForceField.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/BoxStiffSpringForceField.inl
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.h
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALOBJECTINTERACTION_SRC}/BoxStiffSpringForceField.cpp
    ${SOFAGENERALOBJECTINTERACTION_SRC}/InteractionEllipsoidForceField.cpp
    )

sofa_find_package(SofaDeformable REQUIRED)
sofa_find_package(SofaEngine REQUIRED)
sofa_find_package(SofaGeneralEngine REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaDeformable SofaEngine SofaGeneralEngine)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.SolidMechanics.Spring and Sofa.Component.Constraint.Projective instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.SolidMechanics.Spring REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.Spring)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Projective)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
