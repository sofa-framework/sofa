cmake_minimum_required(VERSION 3.12)
project(SceneCreator VERSION 0.1)

sofa_find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Eigen3 REQUIRED)

set(HEADER_FILES
    src/SceneCreator/config.h.in
    src/SceneCreator/GetAssembledSizeVisitor.h
    src/SceneCreator/GetVectorVisitor.h
    src/SceneCreator/SceneCreator.h
    src/SceneCreator/SceneUtils.h
    )
set(SOURCE_FILES
    src/SceneCreator/GetAssembledSizeVisitor.cpp
    src/SceneCreator/GetVectorVisitor.cpp
    src/SceneCreator/SceneCreator.cpp
    src/SceneCreator/SceneUtils.cpp
    src/SceneCreator/initSceneCreator.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Graph Sofa.Component.StateContainer)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SCENECREATOR")

## Install rules for the library; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SceneCreator"
    RELOCATABLE "plugins"
    )

## Add example project to use sceneCreator
sofa_find_package(Sofa.GUI.Common QUIET)
if(Sofa.GUI.Common_FOUND)
    add_subdirectory(sceneCreatorExamples)
else()
    message(STATUS "${PROJECT_NAME}: Sofa.GUI.Common not found, sceneCreatorExamples will not be built.")
endif()

## Add test project
if(SOFA_BUILD_TESTS)
    add_subdirectory(SceneCreator_test)
endif()
