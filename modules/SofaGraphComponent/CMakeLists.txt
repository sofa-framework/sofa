include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initGraphComponent.h
    AddFrameButtonSetting.h
    AddRecordedCameraButtonSetting.h
    AttachBodyButtonSetting.h
    BackgroundSetting.h
    FixPickedParticleButtonSetting.h
    MouseButtonSetting.h
    SofaDefaultPathSetting.h
    StatsSetting.h
    ViewerSetting.h
    Gravity.h
    PauseAnimation.h
    PauseAnimationOnEvent.h
    RequiredPlugin.h
    InteractingBehaviorModel.h

    )

set(SOURCE_FILES

    initGraphComponent.cpp
    AddFrameButtonSetting.cpp
    AddRecordedCameraButtonSetting.cpp
    AttachBodyButtonSetting.cpp
    BackgroundSetting.cpp
    FixPickedParticleButtonSetting.cpp
    MouseButtonSetting.cpp
    SofaDefaultPathSetting.cpp
    StatsSetting.cpp
    ViewerSetting.cpp
    Gravity.cpp
    PauseAnimation.cpp
    PauseAnimationOnEvent.cpp
    RequiredPlugin.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_GRAPH_COMPONENT")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
