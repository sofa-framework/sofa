cmake_minimum_required(VERSION 2.8.12)
set(SOLUTION_NAME "SofaBase")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaBase_ACTUAL_SOURCE_DIR ${SofaBase_SOURCE_DIR}/../modules/)

set(SOFABASE_DEPENDENCY_INCLUDE_DIR)

## SofaBase dependencies
##
find_package(SofaSimulation REQUIRED)
list(APPEND SOFABASE_DEPENDENCY_INCLUDE_DIR ${SofaSimulation_INCLUDE_DIRS})


set(SOFABASE_VERSION ${SofaFramework_VERSION})

configure_file(SofaBaseConfig.h.in SofaBase/SofaBaseConfig.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaBase/SofaBaseConfig.h DESTINATION "include/SofaBase")
# make sure everyone in the build tree can see <SofaBase/SofaBaseConfig.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaBase_ACTUAL_SOURCE_DIR}/../../)

set(SOFABASE_TARGETS
    SofaBaseAnimationLoop
    SofaBaseCollision
    SofaBaseLinearSolver
    SofaBaseMechanics
    SofaBaseTopology
    SofaBaseVisual
    SofaComponentBase
)

foreach(TARGET ${SOFABASE_TARGETS})
    add_subdirectory(${SofaBase_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaBase") # IDE folder
endforeach()

# CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaBase ${SOFABASE_VERSION} "${SofaBase_ACTUAL_SOURCE_DIR}" "include")
