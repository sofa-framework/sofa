cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initGraphComponent.h 
    ../configurationsetting/AddFrameButtonSetting.h 
#   ../configurationsetting/CompliantAttachButtonSetting.h 
    ../configurationsetting/AttachBodyButtonSetting.h 
    ../configurationsetting/BackgroundSetting.h 
    ../configurationsetting/FixPickedParticleButtonSetting.h 
    ../configurationsetting/MouseButtonSetting.h 
    ../configurationsetting/SofaDefaultPathSetting.h 
    ../configurationsetting/StatsSetting.h 
    ../configurationsetting/ViewerSetting.h 
    ../contextobject/Gravity.h 
    ../misc/PauseAnimation.h 
    ../misc/PauseAnimationOnEvent.h 
    ../misc/RequiredPlugin.h

    )
    
set(SOURCE_FILES

    ../initGraphComponent.cpp 
    ../configurationsetting/AddFrameButtonSetting.cpp 
#   ../configurationsetting/CompliantAttachButtonSetting.cpp 
    ../configurationsetting/AttachBodyButtonSetting.cpp 
    ../configurationsetting/BackgroundSetting.cpp 
    ../configurationsetting/FixPickedParticleButtonSetting.cpp 
    ../configurationsetting/MouseButtonSetting.cpp 
    ../configurationsetting/SofaDefaultPathSetting.cpp 
    ../configurationsetting/StatsSetting.cpp 
    ../configurationsetting/ViewerSetting.cpp 
    ../contextobject/Gravity.cpp 
    ../misc/PauseAnimation.cpp 
    ../misc/PauseAnimationOnEvent.cpp 
    ../misc/RequiredPlugin.cpp 

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_GRAPH_COMPONENT")
set(LINKER_DEPENDENCIES SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
