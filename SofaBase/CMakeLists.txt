cmake_minimum_required(VERSION 2.8.12)
set(SOLUTION_NAME "SofaBase")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaBase_ACTUAL_SOURCE_DIR ${SofaBase_SOURCE_DIR}/../modules/)

##
## SofaBase version
##
set(SOFABASE_MAJOR_VERSION 0)
set(SOFABASE_MINOR_VERSION 0)
set(SOFABASE_BUILD_VERSION 1)

set(SOFABASE_VERSION ${SOFABASE_MAJOR_VERSION}.${SOFABASE_MINOR_VERSION}.${SOFABASE_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")


set(SOFABASE_DEPENDENCY_INCLUDE_DIR)

## SofaBase dependencies
##
find_package(SofaSimulation REQUIRED)
list(APPEND SOFABASE_DEPENDENCY_INCLUDE_DIR ${SofaSimulation_INCLUDE_DIRS})

configure_file(SofaBaseConfig.h.in
               SofaBase/SofaBaseConfig.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaBase/SofaBaseConfig.h
         DESTINATION "include/SofaBase")
# make sure everyone in the build tree can see <SofaBase/SofaBaseConfig.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaBase_ACTUAL_SOURCE_DIR}/../../)

set(SOFABASE_TARGETS SofaBaseTopology SofaBaseLinearSolver SofaBaseMechanics SofaBaseCollision
                       SofaBaseAnimationLoop SofaBaseVisual SofaComponentBase)

foreach(TARGET ${SOFABASE_TARGETS})
    add_subdirectory(${SofaBase_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaBase") # IDE folder
endforeach()

# CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaBase ${SOFABASE_VERSION})
