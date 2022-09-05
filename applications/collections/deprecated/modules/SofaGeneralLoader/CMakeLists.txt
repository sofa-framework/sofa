cmake_minimum_required(VERSION 3.12)
project(SofaGeneralLoader LANGUAGES CXX)

set(SOFAGENERALLOADER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALLOADER_SRC}/config.h.in
    ${SOFAGENERALLOADER_SRC}/initSofaGeneralLoader.h
    )

set(SOURCE_FILES
    ${SOFAGENERALLOADER_SRC}/initSofaGeneralLoader.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.IO.Mesh and Sofa.Component.Playback instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.IO.Mesh REQUIRED)
sofa_find_package(Sofa.Component.Playback REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.IO.Mesh)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Playback)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
