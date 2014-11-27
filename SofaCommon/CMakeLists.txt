cmake_minimum_required(VERSION 2.8)
set(SOLUTION_NAME "SofaCommon")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set( SofaCommon_ACTUAL_SOURCE_DIR ${SofaCommon_SOURCE_DIR}/../modules)

##
## SofaCommon version
##
set( SOFACOMMON_MAJOR_VERSION 0 )
set( SOFACOMMON_MINOR_VERSION 0 )
set( SOFACOMMON_BUILD_VERSION 1 )

set( SOFACOMMON_VERSION ${SOFACOMMON_MAJOR_VERSION}.${SOFACOMMON_MINOR_VERSION}.${SOFACOMMON_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SOFACOMMON_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFACOMMON_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFACOMMON_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFACOMMON_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaCommon)

# Offer the user the choice of overriding the installation directories
set(SOFACOMMON_INSTALL_LIB_DIR     lib     					  CACHE PATH "Installation directory for libraries")
set(SOFACOMMON_INSTALL_BIN_DIR     bin     					  CACHE PATH "Installation directory for executables")
set(SOFACOMMON_INSTALL_INCLUDE_DIR include 					  CACHE PATH "Installation directory for header files")
set(SOFACOMMON_INSTALL_CMAKE_DIR   lib/cmake/SofaCommon   CACHE PATH "Installation directory for CMake files")

## SofaCommon dependencies
##
find_package(SofaBase REQUIRED)
set(SOFACOMMON_DEPENDENCY_INCLUDE_DIR ${SofaBase_INCLUDE_DIRS})

configure_file( ${SofaCommon_SOURCE_DIR}/SofaCommon.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaCommon.h )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaCommon.h
         DESTINATION "${SOFACOMMON_INSTALL_INCLUDE_DIR}/sofa")
# make sure everyone in the build tree can see <sofa/SofaCommon.h>
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
#include_directories(${SofaCommon_ACTUAL_SOURCE_DIR})

set(SOFACOMMON_TARGETS SofaLoader SofaRigid SofaDeformable SofaSimpleFem SofaObjectInteraction 
    SofaMeshCollision SofaExplicitOdeSolver SofaImplicitOdeSolver SofaComponentCommon)

foreach(TARGET ${SOFACOMMON_TARGETS})
  add_subdirectory(${SofaCommon_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
  if(MSVC)
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaCommon")
  endif()
endforeach()

include(CMakePackageConfigHelpers)

# SofaCommonConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFACOMMON_BUILD_CMAKE_DIR}/SofaCommonConfigVersion.cmake
                                 VERSION ${SOFACOMMON_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${SOFACOMMON_BUILD_CMAKE_DIR}/SofaCommonConfigVersion.cmake
        DESTINATION ${SOFACOMMON_INSTALL_CMAKE_DIR})

# SofaCommonConfig.cmake (build tree)
set(SOFACOMMON_INCLUDE_DIR "${SOFACOMMON_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaCommonConfig.cmake.in
                              ${SOFACOMMON_BUILD_CMAKE_DIR}/SofaCommonConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFACOMMON_INCLUDE_DIR)

# SofaCommonConfig.cmake (installed)
set(SOFACOMMON_INCLUDE_DIR "${SOFACOMMON_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaCommonConfig.cmake.in
                              ${SOFACOMMON_BUILD_CMAKE_DIR}/SofaCommonConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFACOMMON_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFACOMMON_INCLUDE_DIR )

install(FILES ${SOFACOMMON_BUILD_CMAKE_DIR}/SofaCommonConfigInstall.cmake
        DESTINATION ${SOFACOMMON_INSTALL_CMAKE_DIR}
        RENAME SofaCommonConfig.cmake)

# SofaCommonTargets.cmake (installed)
install(EXPORT SofaCommonTargets
        DESTINATION ${SOFACOMMON_INSTALL_CMAKE_DIR} )
