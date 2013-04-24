cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	SimpleGUI.h
	)

set(SOURCE_FILES

	SimpleGUI.cpp
	)

if(EXTERNAL_HAVE_BOOST)
	list(APPEND HEADER_FILES MultithreadGUI.h)
	
	list(APPEND SOURCE_FILES MultithreadGUI.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${MOC_HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SOFAGUIGLUT")
set(LINKER_DEPENDENCIES SofaGuiCommon ${GLUT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

