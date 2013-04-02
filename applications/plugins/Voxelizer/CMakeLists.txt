cmake_minimum_required(VERSION 2.8)

project("SofaVoxelizer")

include(${SOFA_CMAKE_DIR}/pre.cmake)

##plugin external dependencies
#find_package( SofaVoxelizerLibs REQUIRED)
#include_directories(${SOFAVOXELIZER_INCLUDE_DIRS})

set(HEADER_FILES

    initVoxelizer.h 
    Voxelizer.h 
    Voxelizer.inl 

	)

set(SOURCE_FILES

    initVoxelizer.cpp 
    Voxelizer.cpp 

	)

set(README_FILE Voxelizer.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_VOXELIZER")
set(LINKER_DEPENDENCIES SofaGpuCuda)

include(${SOFA_CMAKE_DIR}/post.cmake)