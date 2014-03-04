include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    NewOmniDriver.h 
    OmniDriver.h
    EnslavementForceFeedback.h
	)

set(SOURCE_FILES

    initSensable.cpp 
    NewOmniDriver.cpp 
    OmniDriver.cpp 
	EnslavementForceFeedback.cpp
	)

set(README_FILES

	PluginSensable.txt
	)

find_package(OPENHAPTICS REQUIRED)
include_directories(${OPENHAPTICS_INCLUDE_DIR})
include_directories(${OPENHAPTICS_UTIL_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_SENSABLEPLUGIN")
AddLinkerDependencies(SofaGuiQt ComponentGeneral BaseCollision Haptics ${OPENHAPTICS_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)