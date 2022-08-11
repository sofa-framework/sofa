cmake_minimum_required(VERSION 3.12)
project(SofaMeshCollision LANGUAGES CXX)

set(SOFAMESHCOLLISION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMESHCOLLISION_SRC}/config.h.in
    ${SOFAMESHCOLLISION_SRC}/fwd.h
    ${SOFAMESHCOLLISION_SRC}/initSofaMeshCollision.h
    ${SOFAMESHCOLLISION_SRC}/RayTriangleIntersection.h
)
set(SOURCE_FILES
    ${SOFAMESHCOLLISION_SRC}/initSofaMeshCollision.cpp
)

sofa_find_package(SofaBaseMechanics REQUIRED)
sofa_find_package(SofaBaseCollision REQUIRED)
sofa_find_package(SofaBaseTopology REQUIRED)
sofa_find_package(SofaObjectInteraction REQUIRED)
sofa_find_package(SofaRigid REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision SofaBaseMechanics SofaBaseTopology)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Collision.Geometry, Sofa.Component.Collision.Detection.Intersection, Sofa.Component.Collision.Response.Mapper and Sofa.Component.Collision.Response.Contact instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Geometry REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Mapper REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Geometry)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Mapper)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
