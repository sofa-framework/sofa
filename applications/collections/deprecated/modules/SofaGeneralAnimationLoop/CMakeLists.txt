cmake_minimum_required(VERSION 3.12)
project(SofaGeneralAnimationLoop LANGUAGES CXX)

set(SOFAGENERALANIMATIONLOOP_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALANIMATIONLOOP_SRC}/config.h.in
    ${SOFAGENERALANIMATIONLOOP_SRC}/initSofaGeneralAnimationLoop.h
    )

set(SOURCE_FILES
    ${SOFAGENERALANIMATIONLOOP_SRC}/initSofaGeneralAnimationLoop.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Mapping.MappedMatrix and Sofa.Component.AnimationLoop instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mapping.MappedMatrix REQUIRED)
sofa_find_package(Sofa.Component.AnimationLoop REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping.MappedMatrix)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.AnimationLoop)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
