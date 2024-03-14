cmake_minimum_required(VERSION 3.12)
project(SofaUserInteraction LANGUAGES CXX)

set(SOFAUSERINTERACTION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAUSERINTERACTION_SRC}/config.h.in
    ${SOFAUSERINTERACTION_SRC}/initSofaUserInteraction.h
    )

set(SOURCE_FILES
    ${SOFAUSERINTERACTION_SRC}/initSofaUserInteraction.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAUSERINTERACTION_SRC}/SleepController.h
    )

list(APPEND SOURCE_FILES
    ${SOFAUSERINTERACTION_SRC}/SleepController.cpp
    )

sofa_find_package(SofaDeformable REQUIRED)
sofa_find_package(SofaGeneralMeshCollision REQUIRED)
sofa_find_package(SofaGeneralVisual REQUIRED)
sofa_find_package(SofaTopologyMapping REQUIRED)
sofa_find_package(SofaBoundaryCondition REQUIRED)
sofa_find_package(SofaGraphComponent REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGeneralMeshCollision SofaGeneralVisual SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Collision.Geometry, Sofa.Component.Collision.Detection.Algorithm, Sofa.Component.Collision.Detection.Intersection, Sofa.Component.Collision.Response.Contact and Sofa.Component.Controller and Sofa.GUI.Component instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Geometry REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
sofa_find_package(Sofa.Component.Controller REQUIRED)
sofa_find_package(Sofa.GUI.Component REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Geometry)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Algorithm)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Controller)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Component)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
