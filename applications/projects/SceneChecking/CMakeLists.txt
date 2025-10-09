cmake_minimum_required(VERSION 3.22)
project(SceneChecking LANGUAGES CXX)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.SceneUtility REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)

set(SCENECHECK_SRC_DIR src/SceneChecking)

set(HEADER_FILES
    ${SCENECHECK_SRC_DIR}/config.h.in
    ${SCENECHECK_SRC_DIR}/init.h
    ${SCENECHECK_SRC_DIR}/SceneCheckAPIChange.h
    ${SCENECHECK_SRC_DIR}/SceneCheckCollisionResponse.h
    ${SCENECHECK_SRC_DIR}/SceneCheckDeprecatedComponents.h
    ${SCENECHECK_SRC_DIR}/SceneCheckDuplicatedName.h
    ${SCENECHECK_SRC_DIR}/SceneCheckEmptyNodeName.h
    ${SCENECHECK_SRC_DIR}/SceneCheckMapping.h
    ${SCENECHECK_SRC_DIR}/SceneCheckMissingRequiredPlugin.h
    ${SCENECHECK_SRC_DIR}/SceneCheckUsingAlias.h
    ${SCENECHECK_SRC_DIR}/SceneCheckerListener.h
    ${SCENECHECK_SRC_DIR}/SceneCheckerVisitor.h
)

set(SOURCE_FILES
    ${SCENECHECK_SRC_DIR}/init.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckAPIChange.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckCollisionResponse.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckDeprecatedComponents.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckDuplicatedName.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckEmptyNodeName.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckMapping.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckMissingRequiredPlugin.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckUsingAlias.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckerListener.cpp
    ${SCENECHECK_SRC_DIR}/SceneCheckerVisitor.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core Sofa.Component.SceneUtility Sofa.Component.Collision.Response.Contact)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SCENECHECKING_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SCENECHECKING_BUILD_TESTS)
    add_subdirectory(tests)
endif()
