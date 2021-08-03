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
    ${SOFAGENERALMESHCOLLISION_SRC}/DirectSAP.h
    ${SOFAGENERALMESHCOLLISION_SRC}/DirectSAPNarrowPhase.h
    ${SOFAGENERALMESHCOLLISION_SRC}/DSAPBox.h
    ${SOFAGENERALMESHCOLLISION_SRC}/IncrSAP.h
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.h
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.inl
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshMinProximityIntersection.h
    ${SOFAGENERALMESHCOLLISION_SRC}/RayTraceNarrowPhase.h
    ${SOFAGENERALMESHCOLLISION_SRC}/TriangleOctree.h
    ${SOFAGENERALMESHCOLLISION_SRC}/TriangleOctreeModel.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALMESHCOLLISION_SRC}/DirectSAP.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/DirectSAPNarrowPhase.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/DSAPBox.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/IncrSAP.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshDiscreteIntersection.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/MeshMinProximityIntersection.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/RayTraceNarrowPhase.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/TriangleOctree.cpp
    ${SOFAGENERALMESHCOLLISION_SRC}/TriangleOctreeModel.cpp
    )

sofa_find_package(SofaBase REQUIRED) # SofaBaseCollision
sofa_find_package(SofaMeshCollision REQUIRED)
sofa_find_package(SofaObjectInteraction REQUIRED)
sofa_find_package(SofaRigid REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaObjectInteraction SofaRigid SofaBaseCollision SofaMeshCollision)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
