cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package( SofaCompliantPluginRunLibs REQUIRED)
include_directories(${SOFACOMPLIANTPLUGINRUN_INCLUDE_DIRS})

set(SOURCE_FILES

    Compliant_run.cpp

	)

if(UNIX)

    set(RC_FILES sofa.rc)

endif()

if(APPLE)

    list(APPEND RC_FILES "runSOFA.incs")

endif()


set(README_FILE  )	

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

set(COMPILER_DEFINES "")
set(LINKER_DEPENDENCIES SofaGuiQt SofaEigen2Solver SofaFlexible)

if(UNIX)
    list(APPEND LINKER_DEPENDECIES "dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)