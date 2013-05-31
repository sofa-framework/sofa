include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package(SofaSensableEmulationLibs REQUIRED)
#include_directories(${SOFASENSABLEEMULATION_INCLUDE_DIRS})

set(HEADER_FILES

    #NewOmniDriverEmu.h 
    OmniDriverEmu.h

	)

set(SOURCE_FILES

    #NewOmniDriverEmu.cpp 
    initSensableEmulation.cpp 
    OmniDriverEmu.cpp

	)

set(README_FILES

	PluginSensableEmulation.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_SENSABLEEMULATIONPLUGIN")
AddLinkerDependencies(SofaGuiQt SofaComponentGeneral SofaHaptics )

include(${SOFA_CMAKE_DIR}/postProject.cmake)