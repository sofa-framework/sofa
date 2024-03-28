cmake_minimum_required(VERSION 3.22)
project(SofaMiscExtra LANGUAGES CXX)

set(SOFAMISCEXTRA_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCEXTRA_SRC}/initSofaMiscExtra.h
    ${SOFAMISCEXTRA_SRC}/config.h.in
    )

set(SOURCE_FILES
    ${SOFAMISCEXTRA_SRC}/initSofaMiscExtra.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Engine.Generate instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Engine.Generate REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Generate)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )
