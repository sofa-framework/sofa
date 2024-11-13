cmake_minimum_required(VERSION 3.22)
project(SofaDistanceGrid.CUDA CUDA CXX)

set(HEADER_FILES
    src/SofaDistanceGrid/CUDA/init.h
    src/SofaDistanceGrid/CUDA/config.h.in

    src/SofaDistanceGrid/CUDA/CudaDistanceGridCollisionModel.h
    src/SofaDistanceGrid/CUDA/CudaDistanceGridContactMapper.h
    src/SofaDistanceGrid/CUDA/CudaCollisionDetection.h
)

set(SOURCE_FILES
    src/SofaDistanceGrid/CUDA/init.cpp

    src/SofaDistanceGrid/CUDA/CudaDistanceGridCollisionModel.cpp
    src/SofaDistanceGrid/CUDA/CudaCollisionDistanceGrid.cpp
    src/SofaDistanceGrid/CUDA/CudaCollisionDetection.cpp
)

set(CUDA_SOURCES
    src/SofaDistanceGrid/CUDA/CudaCollisionDetection.cu
    src/SofaDistanceGrid/CUDA/CudaDistanceGridContactMapper.cu
)

sofa_find_package(SofaDistanceGrid REQUIRED)
sofa_find_package(SofaCUDA REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCES})

target_link_libraries(${PROJECT_NAME} SofaDistanceGrid)
target_link_libraries(${PROJECT_NAME} SofaCUDA)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
