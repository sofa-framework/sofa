cmake_minimum_required(VERSION 3.12)
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
    ${SOFAGRAPHCOMPONENT_SRC}/AddFrameButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/AddRecordedCameraButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/AttachBodyButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/FixPickedParticleButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/Gravity.h
    ${SOFAGRAPHCOMPONENT_SRC}/InteractingBehaviorModel.h
    ${SOFAGRAPHCOMPONENT_SRC}/MouseButtonSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/PauseAnimation.h
    ${SOFAGRAPHCOMPONENT_SRC}/PauseAnimationOnEvent.h
    ${SOFAGRAPHCOMPONENT_SRC}/SofaDefaultPathSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/StatsSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/ViewerSetting.h
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheck.h
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckDuplicatedName.h
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckMissingRequiredPlugin.h
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckAPIChange.h
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckUsingAlias.h
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckerVisitor.h
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckerListener.h
    ${SOFAGRAPHCOMPONENT_SRC}/APIVersion.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGRAPHCOMPONENT_SRC}/AddFrameButtonSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/AddRecordedCameraButtonSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/AttachBodyButtonSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/FixPickedParticleButtonSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/Gravity.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/MouseButtonSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/PauseAnimation.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/PauseAnimationOnEvent.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/SofaDefaultPathSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/StatsSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/ViewerSetting.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckDuplicatedName.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckMissingRequiredPlugin.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckAPIChange.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckUsingAlias.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckerVisitor.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/SceneCheckerListener.cpp
    ${SOFAGRAPHCOMPONENT_SRC}/APIVersion.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaSimulationCore SofaBaseUtils)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAGRAPHCOMPONENT_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGRAPHCOMPONENT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
