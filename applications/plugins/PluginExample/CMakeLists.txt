cmake_minimum_required(VERSION 2.8)

project("SofaPluginExample")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    MyBehaviorModel.h 
    MyDataWidgetUnsigned.h 
    MyMappingPendulumInPlane.h 
    MyMappingPendulumInPlane.inl 
    MyProjectiveConstraintSet.h 
    MyProjectiveConstraintSet.inl 
	initPlugin.h

	)

set(SOURCE_FILES

    MyBehaviorModel.cpp 
    MyDataWidgetUnsigned.cpp 
    MyProjectiveConstraintSet.cpp 
    MyMappingPendulumInPlane.cpp 
    initPlugin.cpp

	)

set(README_FILE PluginExample.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PLUGINEXAMPLE")
set(LINKER_DEPENDENCIES SofaGuiQt)

include(${SOFA_CMAKE_DIR}/postProject.cmake)