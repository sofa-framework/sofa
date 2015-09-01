cmake_minimum_required(VERSION 2.8.11)
set(SOLUTION_NAME "SofaMisc")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaMisc_ACTUAL_SOURCE_DIR ${SofaMisc_SOURCE_DIR}/../modules/)

##
## SofaMisc version
##
set(SOFAMISC_MAJOR_VERSION 0)
set(SOFAMISC_MINOR_VERSION 0)
set(SOFAMISC_BUILD_VERSION 1)

set(SOFAMISC_VERSION ${SOFAMISC_MAJOR_VERSION}.${SOFAMISC_MINOR_VERSION}.${SOFAMISC_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")


## SofaMisc dependencies
##
find_package(SofaGeneral REQUIRED)
set(SOFAMISC_DEPENDENCY_DIR ${SofaGeneral_INCLUDE_DIRS})

configure_file(SofaMiscConfig.h.in
               SofaMisc/SofaMiscConfig.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaMisc/SofaMiscConfig.h
         DESTINATION "include/SofaMisc")
# make sure everyone in the build tree can see <SofaMisc/SofaMiscConfig.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaMisc_ACTUAL_SOURCE_DIR}/../../)

set(SOFAMISC_TARGETS SofaMiscTopology SofaMiscMapping SofaMiscForceField SofaMiscFem SofaMiscEngine
    SofaMiscCollision SofaMiscSolver SofaMisc SofaComponentMisc)

foreach(TARGET ${SOFAMISC_TARGETS})
    add_subdirectory(${SofaMisc_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaMisc") # IDE folder
endforeach()

include(CMakePackageConfigHelpers)

# SofaMiscConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(SofaMiscConfigVersion.cmake
                                 VERSION ${SOFAMISC_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaMiscConfigVersion.cmake
        DESTINATION lib/cmake/SofaMisc)

# SofaMiscConfig.cmake (build tree)
set(SOFAMISC_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SofaMiscConfig.cmake.in
                              SofaMiscConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFAMISC_INCLUDE_DIR)

# SofaMiscConfig.cmake (installed)
set(SOFAMISC_INCLUDE_DIR include)
configure_package_config_file(SofaMiscConfig.cmake.in
                              SofaMiscConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SofaMisc
                              PATH_VARS SOFAMISC_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaMiscConfigInstall.cmake
        DESTINATION lib/cmake/SofaMisc
        RENAME SofaMiscConfig.cmake)

# SofaMiscTargets.cmake (installed)
install(EXPORT SofaMiscTargets
        DESTINATION lib/cmake/SofaMisc)
