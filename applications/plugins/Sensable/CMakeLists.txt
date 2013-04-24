cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
set(LINKER_DEPENDENCIES SofaGuiQt SofaComponentGeneral SofaHaptics )

if(UNIX)
    list(APPEND LINKER_DEPENDENCIES "HD")
    list(APPEND LINKER_DEPENDENCIES "HL")
    list(APPEND LINKER_DEPENDENCIES "HDU")
    list(APPEND LINKER_DEPENDENCIES "HLU")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)