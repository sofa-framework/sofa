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

include(CMakePackageConfigHelpers)

# SofaBaseConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(SofaBaseConfigVersion.cmake
                                 VERSION ${SOFABASE_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaBaseConfigVersion.cmake
        DESTINATION lib/cmake/SofaBase)

# SofaBaseConfig.cmake (build tree)
set(SOFABASE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SofaBaseConfig.cmake.in
                              SofaBaseConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFABASE_INCLUDE_DIR)

# SofaBaseConfig.cmake (installed)
set(SOFABASE_INCLUDE_DIR include)
configure_package_config_file(SofaBaseConfig.cmake.in
                              SofaBaseConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SofaBase
                              PATH_VARS SOFABASE_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaBaseConfigInstall.cmake
        DESTINATION lib/cmake/SofaBase
        RENAME SofaBaseConfig.cmake)

# SofaBaseTargets.cmake (installed)
install(EXPORT SofaBaseTargets
        DESTINATION lib/cmake/SofaBase)
