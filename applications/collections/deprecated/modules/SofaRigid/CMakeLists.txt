cmake_minimum_required(VERSION 3.12)
project(SofaRigid LANGUAGES CXX)

set(SOFARIGID_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFARIGID_SRC}/config.h.in
    ${SOFARIGID_SRC}/initSofaRigid.h
)

set(SOURCE_FILES
    ${SOFARIGID_SRC}/initSofaRigid.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Mapping and Sofa.Component.SolidMechanics.Spring instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mapping REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.Spring REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.Spring)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
