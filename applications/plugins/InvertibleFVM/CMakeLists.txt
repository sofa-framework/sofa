cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    InvertibleFVMForceField.h 
	InvertibleFVMForceField.inl 
	initPlugin.h

	)

set(SOURCE_FILES

    InvertibleFVMForceField.cpp 
    initPlugin.cpp

	)

set(README_FILE InvertibleFVM.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_INVERTIBLEFVM")
set(LINKER_DEPENDENCIES SofaGuiQt )

include(${SOFA_CMAKE_DIR}/postProject.cmake)