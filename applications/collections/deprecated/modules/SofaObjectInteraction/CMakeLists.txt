cmake_minimum_required(VERSION 3.12)
project(SofaObjectInteraction LANGUAGES CXX)

set(SOFAOBJECTINTERACTION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAOBJECTINTERACTION_SRC}/config.h.in
    ${SOFAOBJECTINTERACTION_SRC}/initSofaObjectInteraction.h
)

set(SOURCE_FILES
    ${SOFAOBJECTINTERACTION_SRC}/initSofaObjectInteraction.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Collision.Response.Contact instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
