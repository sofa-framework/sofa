cmake_minimum_required(VERSION 3.22)
project(SofaGeneralVisual LANGUAGES CXX)

set(SOFAGENERALVISUAL_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALVISUAL_SRC}/config.h.in
    ${SOFAGENERALVISUAL_SRC}/initSofaGeneralVisual.h
    )

set(SOURCE_FILES
    ${SOFAGENERALVISUAL_SRC}/initSofaGeneralVisual.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Visual instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Visual REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Visual)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
