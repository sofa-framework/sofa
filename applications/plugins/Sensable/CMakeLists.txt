include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    EnslavementForceFeedback.h
    NewOmniDriver.h
    OmniDriver.h
)

set(SOURCE_FILES
    EnslavementForceFeedback.cpp
    NewOmniDriver.cpp
    OmniDriver.cpp
    initSensable.cpp
)

set(README_FILES PluginSensable.txt)

find_package(OpenHaptics REQUIRED)
include_directories(${OPENHAPTICS_INCLUDE_DIR})
include_directories(${OPENHAPTICS_UTIL_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_SENSABLEPLUGIN")
AddLinkerDependencies(SofaGuiQt SofaComponentGeneral SofaBaseCollision SofaHaptics ${OPENHAPTICS_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)