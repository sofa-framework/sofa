cmake_minimum_required(VERSION 3.22)
project(SofaGraphComponent LANGUAGES CXX)

sofa_find_package(SofaBase REQUIRED)

set(SOFAGRAPHCOMPONENT_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGRAPHCOMPONENT_SRC}/config.h.in
    ${SOFAGRAPHCOMPONENT_SRC}/initSofaGraphComponent.h
    )

set(SOURCE_FILES
    ${SOFAGRAPHCOMPONENT_SRC}/initSofaGraphComponent.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGRAPHCOMPONENT_SRC}/Gravity.h
    ${SOFAGRAPHCOMPONENT_SRC}/InteractingBehaviorModel.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGRAPHCOMPONENT_SRC}/Gravity.cpp
    )


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper Sofa.Simulation.Core SofaBaseUtils SofaBaseCollision)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.SceneUtility, Sofa.Component.Setting,
#Sofa.GUI.Component and the plugin SceneChecking instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.SceneUtility REQUIRED)
sofa_find_package(Sofa.Component.Setting REQUIRED)
sofa_find_package(Sofa.GUI.Component REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SceneUtility)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Setting)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Component)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
