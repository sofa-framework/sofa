cmake_minimum_required(VERSION 2.8)

project("SofaSensableEmulation")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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

set(README_FILE PluginSensableEmulation.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SENSABLEEMULATIONPLUGIN")
set(LINKER_DEPENDENCIES SofaGuiQt SofaComponentGeneral SofaHaptics )

include(${SOFA_CMAKE_DIR}/post.cmake)