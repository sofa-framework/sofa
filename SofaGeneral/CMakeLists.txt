cmake_minimum_required(VERSION 2.8.12)
project(SofaGeneral)

## Options
option(SOFAGENERAL_PARDISOSOLVER "Build the SofaPardisoSolver library" OFF)


## Dependencies
find_package(SofaCommon REQUIRED)
find_package(Newmat)
set(SOFA_HAVE_NEWMAT ${Newmat_FOUND})
find_package(CSparse)
set(SOFA_HAVE_CSPARSE ${CSparse_FOUND})
set(SOFAGENERAL_DEPENDENCY_INCLUDE_DIR ${SofaCommon_INCLUDE_DIRS})

## Version
set(SOFAGENERAL_VERSION ${SofaFramework_VERSION})

## SofaGeneral/config.h
configure_file(config.h.in SofaGeneral/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaGeneral/config.h DESTINATION "include/SofaGeneral")

## Installation macro for subdirectories
macro(SofaGeneral_install lib)
    install(TARGETS ${lib}
            EXPORT SofaGeneralTargets
            RUNTIME DESTINATION bin COMPONENT SofaGeneral_libraries
            LIBRARY DESTINATION lib COMPONENT SofaGeneral_libraries
            ARCHIVE DESTINATION lib COMPONENT SofaGeneral_libraries
            PUBLIC_HEADER DESTINATION include/${lib} COMPONENT SofaGeneral_headers)
endmacro()


### Subdirectories

set(SOFAGENERAL_TARGETS
    SofaBoundaryCondition
    SofaComponentGeneral
    SofaConstraint
    SofaEngine
    SofaExporter
    SofaGraphComponent
    SofaHaptics
    SofaPreconditioner
    SofaSparseSolver
    SofaTopologyMapping
    SofaUserInteraction
    SofaValidation
)

if(SOFA_HAVE_NEWMAT)
    list(APPEND SOFAGENERAL_TARGETS SofaDenseSolver)
endif()

if(NOT SOFA_NO_OPENGL)
    list(APPEND SOFAGENERAL_TARGETS SofaOpenglVisual)
endif()

if(SOFA_HAVE_TAUCS)
    list(APPEND SOFAGENERAL_TARGETS SofaTaucsSolver)
endif()

if(SOFA_HAVE_EIGEN2)
    list(APPEND SOFAGENERAL_TARGETS SofaEigen2Solver)
endif()

if(SOFAGENERAL_PARDISOSOLVER)
    list(APPEND SOFAGENERAL_TARGETS SofaPardisoSolver)
endif()

foreach(TARGET ${SOFAGENERAL_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>") # For SofaGeneral/config.h
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${SofaCommon_INCLUDE_DIRS}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaGeneral) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAGENERAL_VERSION})
endforeach()


## CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaGeneral ${SOFAGENERAL_VERSION} "" "include")
