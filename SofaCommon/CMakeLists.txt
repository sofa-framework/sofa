cmake_minimum_required(VERSION 2.8)
set(SOLUTION_NAME "SofaCommon")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaCommon_ACTUAL_SOURCE_DIR ${SofaCommon_SOURCE_DIR}/../modules)

## SofaCommon dependencies
##
find_package(SofaBase REQUIRED)
set(SOFACOMMON_DEPENDENCY_INCLUDE_DIR ${SofaBase_INCLUDE_DIRS})


set(SOFACOMMON_VERSION ${SofaFramework_VERSION})

configure_file(config.h.in SofaCommon/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaCommon/config.h DESTINATION "include/SofaCommon")
# make sure everyone in the build tree can see <SofaCommon/config.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaCommon_ACTUAL_SOURCE_DIR})

set(SOFACOMMON_TARGETS
    SofaComponentCommon
    SofaDeformable
    SofaExplicitOdeSolver
    SofaImplicitOdeSolver
    SofaLoader
    SofaMeshCollision
    SofaObjectInteraction
    SofaRigid
    SofaSimpleFem
)

foreach(TARGET ${SOFACOMMON_TARGETS})
    add_subdirectory(${SofaCommon_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaCommon") # IDE folder
endforeach()

# CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaCommon ${SOFACOMMON_VERSION} "" "include")
