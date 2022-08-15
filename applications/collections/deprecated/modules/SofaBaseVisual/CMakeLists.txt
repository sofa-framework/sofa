cmake_minimum_required(VERSION 3.12)
project(SofaBaseVisual LANGUAGES CXX)

set(SOFABASEVISUAL_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASEVISUAL_SRC}/config.h.in
    ${SOFABASEVISUAL_SRC}/initSofaBaseVisual.h
)

set(SOURCE_FILES
    ${SOFABASEVISUAL_SRC}/initSofaBaseVisual.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Visual and Sofa.Component.Setting instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Visual REQUIRED)
sofa_find_package(Sofa.Component.Setting REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Visual)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Setting)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
