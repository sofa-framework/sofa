cmake_minimum_required(VERSION 3.12)
project(SofaMiscMapping LANGUAGES CXX)

set(SOFAMISCMAPPING_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCMAPPING_SRC}/config.h.in
    ${SOFAMISCMAPPING_SRC}/initSofaMiscMapping.h
    )

set(SOURCE_FILES
    ${SOFAMISCMAPPING_SRC}/initSofaMiscMapping.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Mapping.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mapping REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )
