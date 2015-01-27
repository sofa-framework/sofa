cmake_minimum_required(VERSION 2.8)
project(Voxelizer)

set(VOXELIZER_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(VOXELIZER_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(VOXELIZER_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    initVoxelizer.h 
    Voxelizer.h 
    Voxelizer.inl)

set(SOURCE_FILES
    initVoxelizer.cpp 
    Voxelizer.cpp)

set(README_FILES
	Voxelizer.txt)

find_package(SofaFramework REQUIRED)
find_package(SofaCUDALDI REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VOXELIZER")
target_link_libraries(${PROJECT_NAME} SofaCore SofaCUDALDI)

install(TARGETS ${PROJECT_NAME}
    COMPONENT Voxelizer_libraries
    EXPORT VoxelizerTargets
    RUNTIME DESTINATION "${VOXELIZER_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${VOXELIZER_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${VOXELIZER_INSTALL_LIB_DIR}")
