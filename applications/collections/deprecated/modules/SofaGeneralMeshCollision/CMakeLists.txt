cmake_minimum_required(VERSION 3.12)
project(SofaGeneralMeshCollision LANGUAGES CXX)

set(SOFAGENERALMESHCOLLISION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALMESHCOLLISION_SRC}/config.h.in
    ${SOFAGENERALMESHCOLLISION_SRC}/initSofaGeneralMeshCollision.h
    )

set(SOURCE_FILES
    ${SOFAGENERALMESHCOLLISION_SRC}/initSofaGeneralMeshCollision.cpp
    )


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Collision.Geometry, Sofa.Component.Collision.Detection.Algorithm and Sofa.Component.Collision.Detection.Intersection instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Geometry REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Geometry)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Algorithm)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
