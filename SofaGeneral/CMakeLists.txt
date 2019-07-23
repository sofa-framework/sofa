cmake_minimum_required(VERSION 3.1)
project(SofaGeneral)

# Send warnings when deprecated CMake project is activated
option(CMAKE_WARN_DEPRECATED "Send warning when deprecated projects are found" ON)

## Dependencies
find_package(SofaCommon REQUIRED)
find_package(Newmat)
set(SOFA_HAVE_NEWMAT ${Newmat_FOUND})

## Version
set(SOFAGENERAL_VERSION ${SofaFramework_VERSION})

## SofaGeneral/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h" DESTINATION "include/${PROJECT_NAME}/${PROJECT_NAME}" COMPONENT headers)

### Subdirectories

set(SOFAGENERAL_TARGETS
    SofaGeneralAnimationLoop
    SofaGeneralDeformable
    SofaGeneralExplicitOdeSolver
    SofaGeneralImplicitOdeSolver
    SofaGeneralLinearSolver
    SofaGeneralLoader
    SofaGeneralMeshCollision
    SofaGeneralObjectInteraction
    SofaGeneralRigid
    SofaGeneralSimpleFem
    SofaGeneralTopology
    SofaGeneralVisual
    SofaBoundaryCondition
    SofaConstraint
    SofaGeneralEngine
    SofaGraphComponent
    SofaTopologyMapping
    SofaUserInteraction
    SofaValidation
)

if(SOFA_HAVE_NEWMAT)
    list(APPEND SOFAGENERAL_TARGETS SofaDenseSolver)
endif()

foreach(TARGET ${SOFAGENERAL_TARGETS} SofaComponentGeneral)
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaGeneral) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAGENERAL_VERSION})
endforeach()


## CMake package configurations files
sofa_write_package_config_files(SofaGeneral ${SOFAGENERAL_VERSION})
