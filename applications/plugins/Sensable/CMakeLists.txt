cmake_minimum_required(VERSION 2.8)

project("SofaSensable")

include(${SOFA_CMAKE_DIR}/pre.cmake)

##plugin external dependencies
find_package(SofaSensableLibs REQUIRED)
include_directories(${SOFASENSABLE_INCLUDE_DIRS})

set(HEADER_FILES

    NewOmniDriver.h 
    OmniDriver.h

	)

set(SOURCE_FILES

    initSensable.cpp 
    NewOmniDriver.cpp 
    OmniDriver.cpp 

	)

set(README_FILE PluginSensable.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SENSABLEPLUGIN")
set(LINKER_DEPENDENCIES SofaGUIQt SofaComponentGeneral SofaHaptics )

include(${SOFA_CMAKE_DIR}/post.cmake)