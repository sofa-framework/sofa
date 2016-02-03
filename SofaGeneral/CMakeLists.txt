cmake_minimum_required(VERSION 2.8.12)
project(SofaGeneral)



## Dependencies
find_package(SofaCommon REQUIRED)
find_package(Newmat)
set(SOFA_HAVE_NEWMAT ${Newmat_FOUND})
find_package(CSparse)
set(SOFA_HAVE_CSPARSE ${CSparse_FOUND})
find_package(Metis QUIET)
set(SOFA_HAVE_METIS ${Metis_FOUND})

## Version
set(SOFAGENERAL_VERSION ${SofaFramework_VERSION})

## SofaGeneral/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaGeneral/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/SofaGeneral/config.h" DESTINATION "include/SofaGeneral")

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

list(APPEND SOFAGENERAL_TARGETS SofaEigen2Solver)


foreach(TARGET ${SOFAGENERAL_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaGeneral) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAGENERAL_VERSION})
endforeach()


## CMake package configurations files
sofa_write_package_config_files(SofaGeneral ${SOFAGENERAL_VERSION})
