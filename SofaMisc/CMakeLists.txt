cmake_minimum_required(VERSION 2.8.11)
set(SOLUTION_NAME "SofaMisc")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaMisc_ACTUAL_SOURCE_DIR ${SofaMisc_SOURCE_DIR}/../modules/)

## SofaMisc dependencies
##
find_package(SofaGeneral REQUIRED)
set(SOFAMISC_DEPENDENCY_DIR ${SofaGeneral_INCLUDE_DIRS})

configure_file(SofaMiscConfig.h.in SofaMisc/SofaMiscConfig.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaMisc/SofaMiscConfig.h DESTINATION "include/SofaMisc")
# make sure everyone in the build tree can see <SofaMisc/SofaMiscConfig.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaMisc_ACTUAL_SOURCE_DIR}/../../)

set(SOFAMISC_TARGETS
    SofaComponentMisc
    SofaMisc
    SofaMiscCollision
    SofaMiscEngine
    SofaMiscFem
    SofaMiscForceField
    SofaMiscMapping
    SofaMiscSolver
    SofaMiscTopology
)

foreach(TARGET ${SOFAMISC_TARGETS})
    add_subdirectory(${SofaMisc_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaMisc") # IDE folder
endforeach()

# CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaMisc ${SofaFramework_VERSION})
