cmake_minimum_required(VERSION 2.8.12)
project(SofaPardisoSolver)

set(SOFAPARDISOSOLVER_MAJOR_VERSION 0)
set(SOFAPARDISOSOLVER_MINOR_VERSION 1)
set(SOFAPARDISOSOLVER_VERSION ${SOFAPARDISOSOLVER_MAJOR_VERSION}.${SOFAPARDISOSOLVER_MINOR_VERSION})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(SofaFramework REQUIRED)
find_package(SofaBase REQUIRED)

link_directories("${CMAKE_CURRENT_SOURCE_DIR}/extlibs")
set(SOFAPARDISOSOLVER_LINK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extlibs")

find_package(GFortran REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

set(LINKER_DEPENDENCIES ${LINKER_DEPENDENCIES} SofaComponentBase SofaComponentGeneral)
set(COMPILATION_FLAGS -DSOFA_BUILD_SOFAPARDISOSOLVER )



set(HEADER_FILES
    SparsePARDISOSolver.h
    config.h
)

set(SOURCE_FILES
    SparsePARDISOSolver.cpp
    initPardisoSolver.cpp
)


set(LINKER_DEPENDENCIES ${LINKER_DEPENDENCIES} SofaSimulationTree SofaMeshCollision SofaBaseVisual SofaBaseLinearSolver SofaCore)
set(LINKER_DEPENDENCIES ${LINKER_DEPENDENCIES} pardiso500-GNU461-X86-64 gfortran blas lapack gomp)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAPARDISOSOLVER_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILATION_FLAGS}")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_link_libraries(${PROJECT_NAME} ${LINKER_DEPENDENCIES})


## Install rules for the library; CMake package configurations files
sofa_create_package(SofaPardisoSolver ${SOFAPARDISOSOLVER_VERSION} SofaPardisoSolver SofaPardisoSolver)