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

list(APPEND HEADER_FILES

    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.h
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.inl
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshMinProximityIntersection.h
    )
list(APPEND SOURCE_FILES

    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshMinProximityIntersection.cpp
    )

sofa_find_package(SofaBase REQUIRED) # SofaBaseCollision
sofa_find_package(SofaMeshCollision REQUIRED)
sofa_find_package(SofaObjectInteraction REQUIRED)
sofa_find_package(SofaRigid REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision SofaMeshCollision)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Collision.Model and Sofa.Component.Collision.Detection.Algorithm instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Model REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Algorithm)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
