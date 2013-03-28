cmake_minimum_required(VERSION 2.8)

project("SofaCompliantPluginRun")

include(${SOFA_CMAKE_DIR}/pre.cmake)

##plugin external dependencies
find_package( SofaCompliantPluginRunLibs REQUIRED)
include_directories(${SOFACOMPLIANTPLUGINRUN_INCLUDE_DIRS})

set(SOURCE_FILES

    Compliant_run.cpp

	)

set(README_FILE  )	

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

set(COMPILER_DEFINES "")
set(LINKER_DEPENDENCIES SofaGuiQt SofaEigen2Solver SofaFlexible)

include(${SOFA_CMAKE_DIR}/post.cmake)