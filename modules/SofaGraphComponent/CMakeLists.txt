cmake_minimum_required(VERSION 3.1)
project(SofaGraphComponent)

set(HEADER_FILES
    config.h
    initGraphComponent.h
    )

set(SOURCE_FILES
    initGraphComponent.cpp
    )

list(APPEND HEADER_FILES
    AddFrameButtonSetting.h
    AddRecordedCameraButtonSetting.h
    AttachBodyButtonSetting.h
    BackgroundSetting.h
    FixPickedParticleButtonSetting.h
    Gravity.h
    InteractingBehaviorModel.h
    MouseButtonSetting.h
    PauseAnimation.h
    PauseAnimationOnEvent.h
    RequiredPlugin.h
    SofaDefaultPathSetting.h
    StatsSetting.h
    ViewerSetting.h
    SceneChecks.h
    SceneCheckAPIChanges.h
    SceneCheckerVisitor.h
    APIVersion.h
    )
list(APPEND SOURCE_FILES
    AddFrameButtonSetting.cpp
    AddRecordedCameraButtonSetting.cpp
    AttachBodyButtonSetting.cpp
    BackgroundSetting.cpp
    FixPickedParticleButtonSetting.cpp
    Gravity.cpp
    MouseButtonSetting.cpp
    PauseAnimation.cpp
    PauseAnimationOnEvent.cpp
    RequiredPlugin.cpp
    SofaDefaultPathSetting.cpp
    StatsSetting.cpp
    ViewerSetting.cpp
    SceneChecks.cpp
    SceneCheckAPIChanges.cpp
    SceneCheckerVisitor.cpp
    APIVersion.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaSimulationCore SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GRAPH_COMPONENT")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
