cmake_minimum_required(VERSION 3.22)
project(SofaLoader LANGUAGES CXX)

set(SOFALOADER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFALOADER_SRC}/config.h.in
    ${SOFALOADER_SRC}/initSofaLoader.h
)

set(SOURCE_FILES
    ${SOFALOADER_SRC}/initSofaLoader.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.IO.Mesh instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.IO.Mesh REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.IO.Mesh)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
