cmake_minimum_required(VERSION 3.12)
project(SofaMiscEngine LANGUAGES CXX)

sofa_find_package(SofaNonUniformFem REQUIRED)

set(SOFAMISCENGINE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCENGINE_SRC}/config.h.in
    ${SOFAMISCENGINE_SRC}/initSofaMiscEngine.h
    )

set(SOURCE_FILES
    ${SOFAMISCENGINE_SRC}/initSofaMiscEngine.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Engine.Analyze and Sofa.Component.Engine.Transform instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Engine.Analyze REQUIRED)
sofa_find_package(Sofa.Component.Engine.Transform REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Analyze)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Engine.Transform)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )
