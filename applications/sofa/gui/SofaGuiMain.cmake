cmake_minimum_required(VERSION 2.8)

project("SofaGuiMain")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

	Main.h
	)

set(SOURCE_FILES

	Main.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_GUIMAIN")
set(LINKER_DEPENDENCIES SofaGuiCommon)

if(GUI_USE_QTVIEWER)
	UseQt()
    list(APPEND LINKER_DEPENDENCIES SofaGuiQt)
endif()

if(GUI_USE_GLUT)
    list(APPEND LINKER_DEPENDENCIES SofaGuiGlut)
endif()

include(${SOFA_CMAKE_DIR}/post.cmake)

