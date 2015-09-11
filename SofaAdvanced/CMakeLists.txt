cmake_minimum_required(VERSION 2.8.11)
set(SOLUTION_NAME "SofaAdvanced")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaAdvanced_ACTUAL_SOURCE_DIR ${SofaAdvanced_SOURCE_DIR}/../modules/)

## SofaAdvanced dependencies
##
find_package(SofaGeneral REQUIRED)
find_package(MiniFlowVR)

set(SOFA_HAVE_MINIFLOWVR ${MiniFlowVR_FOUND})

set(SOFAADVANCED_DEPENDENCY_DIR ${SofaGeneral_INCLUDE_DIRS} ${MiniFlowVR_INCLUDE_DIRS})
set(SOFAADVANCED_DEPENDENCY_LINK ${MiniFlowVR_LIBRARIES})


set(SOFAADVANCED_VERSION ${SofaFramework_VERSION})

configure_file(SofaAdvancedConfig.h.in SofaAdvanced/SofaAdvancedConfig.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaAdvanced/SofaAdvancedConfig.h DESTINATION "include/SofaAdvanced")
# make sure everyone in the build tree can see <SofaAdvanced/SofaAdvancedConfig.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaAdvanced_ACTUAL_SOURCE_DIR}/../../)

set(SOFAADVANCED_TARGETS
    SofaComponentAdvanced
    SofaEulerianFluid
    SofaNonUniformFem
    SofaSphFluid
    SofaVolumetricData
)

foreach(TARGET ${SOFAADVANCED_TARGETS})
    add_subdirectory(${SofaAdvanced_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaAdvanced") # IDE folder
endforeach()

# CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaAdvanced ${SofaFramework_VERSION})
