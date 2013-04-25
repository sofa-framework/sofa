cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	../Main.h
	)

set(SOURCE_FILES

	../Main.cpp
	)
	
set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_GUIMAIN")
set(LINKER_DEPENDENCIES SofaGuiCommon)

if(SOFA-LIB_GUI_QTVIEWER)
	UseQt()
    list(APPEND LINKER_DEPENDENCIES SofaGuiQt)
endif()

if(SOFA-LIB_GUI_GLUT)
    list(APPEND LINKER_DEPENDENCIES SofaGuiGlut)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

