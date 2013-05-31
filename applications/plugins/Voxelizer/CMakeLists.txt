include(${SOFA_CMAKE_DIR}/preProject.cmake)

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

set(README_FILES

	Voxelizer.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_VOXELIZER")
AddLinkerDependencies(SofaCUDALDI)

include(${SOFA_CMAKE_DIR}/postProject.cmake)