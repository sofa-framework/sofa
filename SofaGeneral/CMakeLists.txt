cmake_minimum_required(VERSION 2.8)
set(SOLUTION_NAME "SofaGeneral")
project(${SOLUTION_NAME})


#temporary until we move the sub projects
set(SofaGeneral_ACTUAL_SOURCE_DIR ${SofaGeneral_SOURCE_DIR}/../modules)


## SofaGeneral dependencies
##
find_package(SofaCommon REQUIRED)
find_package(Newmat)
set(SOFA_HAVE_NEWMAT ${Newmat_FOUND})
find_package(CSparse)
set(SOFA_HAVE_CSPARSE ${CSparse_FOUND})

set(SOFAGENERAL_DEPENDENCY_INCLUDE_DIR ${SofaCommon_INCLUDE_DIRS})


set(SOFAGENERAL_VERSION ${SofaFramework_VERSION})

configure_file(config.h.in SofaGeneral/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaGeneral/config.h DESTINATION "include/SofaGeneral")

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

option(SOFAGENERAL_PARDISOSOLVER "Build the SofaPardisoSolver library" OFF)
if(SOFAGENERAL_PARDISOSOLVER)
    list(APPEND SOFAGENERAL_TARGETS SofaPardisoSolver)
endif()

foreach(TARGET ${SOFAGENERAL_TARGETS})
    add_subdirectory(${SofaGeneral_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaGeneral") # IDE folder
endforeach()

# CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaGeneral ${SOFAGENERAL_VERSION} "" "include")
