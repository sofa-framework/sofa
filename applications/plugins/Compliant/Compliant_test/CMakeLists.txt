cmake_minimum_required(VERSION 2.8)

project("SofaCompliantPluginTest")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package( SofaCompliantPluginTestLibs REQUIRED)
include_directories(${SOFACOMPLIANTPLUGINTEST_INCLUDE_DIRS})

set(SOURCE_FILES

    Compliant_test.cpp

	)

if(APPLE)

    set(RC_FILES "runSOFA.icns")

endif()

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

set(COMPILER_DEFINES "")
set(LINKER_DEPENDENCIES SofaGuiQt SofaEigen2Solver SofaFlexible)

if(UNIX)
    list(APPEND LINKER_DEPENDENCIES "dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)