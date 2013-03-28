cmake_minimum_required(VERSION 2.8)

project("SofaXitact")

include(${SOFA_CMAKE_DIR}/pre.cmake)

##plugin external dependencies
find_package( SofaXitactLibs REQUIRED)
include_directories(${SOFAXITACT_INCLUDE_DIRS})

set(HEADER_FILES

    XiTrocarInterface.h 
    initXitact.h 
    PaceMaker.h 
    IHPDriver.h
    ITPDriver.h 

	)

set(SOURCE_FILES

    initXitact.cpp 
    PaceMaker.cpp  
    IHPDriver.cpp 
    ITPDriver.cpp 

	)

set(README_FILE PluginXitact.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_XITACTPLUGIN")
set(LINKER_DEPENDENCIES SofaComponentMain SofaGui SofaGuiQt)

include(${SOFA_CMAKE_DIR}/post.cmake)