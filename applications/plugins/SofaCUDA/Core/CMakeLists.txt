cmake_minimum_required(VERSION 3.22)
project(SofaCUDA.Core LANGUAGES CUDA CXX)

set(SOFACUDA_CORE_MAJOR_VERSION 0)
set(SOFACUDA_CORE_MINOR_VERSION 1)
set(SOFACUDA_CORE_VERSION ${SOFACUDA_CORE_MAJOR_VERSION}.${SOFACUDA_CORE_MINOR_VERSION})

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0") 
    option(SOFACUDA_ENABLE_NATIVE_ARCHITECTURE "Set native as for CUDA_ARCHITECTURES (which will compile compatible architectures for the current system)")
    if(SOFACUDA_ENABLE_NATIVE_ARCHITECTURE)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

# set 75 as fallback value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(NOTICE "CMAKE_CUDA_ARCHITECTURES is not set, it will be set by default to 75")
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(SOFACUDA_CORE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(HEADER_FILES
    ${SOFACUDA_CORE_SOURCE_DIR}/SofaCUDA/core/config.h.in
    ${SOFACUDA_CORE_SOURCE_DIR}/SofaCUDA/core/init.h
    
    ### Core
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaBaseVector.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaCommon.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMath.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMath.inl
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMathRigid.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMathRigid.inl
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMatrix.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMemoryManager.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaScan.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSort.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaTypes.h
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/mycuda.h

    ### Mechanical
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSingleStateAccessor.h
)

set(SOURCE_FILES
    ${SOFACUDA_CORE_SOURCE_DIR}/SofaCUDA/core/init.cpp

    ### Core
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaBaseVector.cpp
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/mycuda.cpp
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSingleStateAccessor.cpp
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMultiMapping.cpp
)

set(CUDA_SOURCES
    
    ### Core
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/mycuda.cu
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaBaseVector.cu
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaScan.cu
    ${SOFACUDA_CORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSort.cu    
)

sofa_find_package(Sofa.GL REQUIRED) # mandatory for gl_interop, buffers....
sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Mass REQUIRED) # because of CudaMassType -> needs MassType (which should probably be moved into Sofa.Core)

option(SOFACUDA_CORE_VERBOSE_PTXAS "???" OFF)

option(SOFACUDA_CUBLAS "Activate cublas support in CUDA (requires SOFACUDA_DOUBLE)." OFF)
if(SOFACUDA_CUBLAS)
    set(SOFA_GPU_CUBLAS 1)       # #cmakedefine
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    find_package(CUDASparse REQUIRED)
endif()

# Note: THRUST is included in CUDA SDK 4.0+, it is recommended to use it if available
option(SOFACUDA_THRUST "Activate THRUST (for RadixSort)." ON)
if(SOFACUDA_THRUST)
    set(SOFA_GPU_THRUST 1)       # #cmakedefine
endif()

option(SOFACUDA_DOUBLE "Activate double-precision support in CUDA (requires GT200+ GPU and -arch sm_13 flag." OFF)
if(SOFACUDA_DOUBLE)
    set(SOFA_GPU_CUDA_DOUBLE 1)       # #cmakedefine
endif()


# Note: with SOFA_GPU_CUDA_PRECISE and SOFA_GPU_CUDA_DOUBLE you get IEEE
# 754-compliant floating point operations for addition and multiplication only.
option(SOFACUDA_PRECISE "Use IEEE 754-compliant floating point operations." OFF)
option(SOFACUDA_DOUBLE_PRECISE "Enable double-precision for sqrt/div..." OFF)
if(SOFACUDA_PRECISE)
    set(SOFA_GPU_CUDA_PRECISE 1)       # #cmakedefine
endif()
if(SOFACUDA_DOUBLE_PRECISE)
    set(SOFA_GPU_CUDA_DOUBLE_PRECISE 1)       # #cmakedefine
endif()


sofa_find_package(CUDAToolkit 11.0 REQUIRED)

# nvcc uses a "host code compiler" to compile CPU code, specified by CUDA_HOST_COMPILER.
# With some versions of CMake, CUDA_HOST_COMPILER defaults to CMAKE_C_COMPILER,
# but few host compilers are actually supported. Workarounds should go here.
if (${CUDA_HOST_COMPILER} MATCHES "ccache$")
    message(STATUS "SofaCUDA: CUDA host compiler was set to ccache, changing to g++")
    set(CUDA_HOST_COMPILER "g++" CACHE STRING "Host side compiler used by NVCC" FORCE)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACUDA_COMPILE_DEFINITIONS}")
if(SOFACUDA_VERBOSE_PTXAS)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()
if(MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:__cplusplus>)
endif()
if(WIN32)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-DWIN32>)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mass)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17 cuda_std_17)

if(SOFACUDA_CUBLAS)
    target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY} CUDA::cublas)
endif()

# see (I guess) https://github.com/sofa-framework/sofa/blob/314b95cbfba411bf8431486ea75b7c67c0bbcc76/Sofa/framework/Config/cmake/SofaMacrosInstall.cmake#L695
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>/include/${PROJECT_NAME}/${PROJECT_NAME}")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFACUDA_CORE_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
)
