cmake_minimum_required(VERSION 2.8)
set(SOLUTION_NAME "SofaGeneral")
project(${SOLUTION_NAME})


#temporary until we move the sub projects
set( SofaGeneral_ACTUAL_SOURCE_DIR ${SofaGeneral_SOURCE_DIR}/../modules)

##
## SofaGeneral version
##
set( SOFAGENERAL_MAJOR_VERSION 0 )
set( SOFAGENERAL_MINOR_VERSION 0 )
set( SOFAGENERAL_BUILD_VERSION 1 )

set( SOFAGENERAL_VERSION ${SOFAGENERAL_MAJOR_VERSION}.${SOFAGENERAL_MINOR_VERSION}.${SOFAGENERAL_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SOFAGENERAL_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFAGENERAL_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFAGENERAL_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAGENERAL_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaGeneral)

# Offer the user the choice of overriding the installation directories
set(SOFAGENERAL_INSTALL_LIB_DIR     lib                     CACHE PATH "Installation directory for libraries")
set(SOFAGENERAL_INSTALL_BIN_DIR     bin                     CACHE PATH "Installation directory for executables")
set(SOFAGENERAL_INSTALL_INCLUDE_DIR include                 CACHE PATH "Installation directory for header files")
set(SOFAGENERAL_INSTALL_CMAKE_DIR   ${SOFAGENERAL_INSTALL_LIB_DIR}/cmake/SofaGeneral )

## SofaGeneral dependencies
##
find_package(SofaCommon REQUIRED)
find_package(Newmat)
set(SOFA_HAVE_NEWMAT ${Newmat_FOUND} )

set(SOFAGENERAL_DEPENDENCY_INCLUDE_DIR ${SofaCommon_INCLUDE_DIRS})


configure_file( ${SofaGeneral_SOURCE_DIR}/SofaGeneral.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaGeneral.h )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaGeneral.h
         DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/sofa")

set(SOFAGENERAL_TARGETS SofaValidation SofaExporter SofaGraphComponent SofaTopologyMapping SofaBoundaryCondition 
    SofaComponentGeneral SofaEngine SofaUserInteraction SofaConstraint SofaHaptics SofaPreconditioner)

if(SOFA_HAVE_NEWMAT)
  list(APPEND SOFAGENERAL_DEPENDENCY_LINK newmat )
  list(APPEND SOFAGENERAL_TARGETS SofaDenseSolver)
endif()

if(NOT SOFA-MISC_NO_OPENGL)
   list(APPEND SOFAGENERAL_TARGETS SofaOpenglVisual)
endif()
#if(SOFA_HAVE_CSPARSE)
   list(APPEND SOFAGENERAL_TARGETS SofaSparseSolver)
#endif()
if(SOFA_HAVE_TAUCS)
   list(APPEND SOFAGENERAL_TARGETS SofaTaucsSolver)
endif()
if(SOFA_HAVE_EIGEN2)
   list(APPEND SOFAGENERAL_TARGETS SofaEigen2Solver)
endif()

foreach(TARGET ${SOFAGENERAL_TARGETS})
  add_subdirectory(${SofaGeneral_ACTUAL_SOURCE_DIR}/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
  # set IDE project filter
  if(MSVC)
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaGeneral")
  endif()
endforeach()

include(CMakePackageConfigHelpers)

# SofaGeneralConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFAGENERAL_BUILD_CMAKE_DIR}/SofaGeneralConfigVersion.cmake
                                 VERSION ${SOFAGENERAL_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${SOFAGENERAL_BUILD_CMAKE_DIR}/SofaGeneralConfigVersion.cmake
        DESTINATION ${SOFAGENERAL_INSTALL_CMAKE_DIR})

# SofaGeneralConfig.cmake (build tree)
set(SOFAGENERAL_INCLUDE_DIR "${SOFAGENERAL_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaGeneralConfig.cmake.in
                              ${SOFAGENERAL_BUILD_CMAKE_DIR}/SofaGeneralConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFAGENERAL_INCLUDE_DIR)

# SofaGeneralConfig.cmake (installed)
set(SOFAGENERAL_INCLUDE_DIR "${SOFAGENERAL_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaGeneralConfig.cmake.in
                              ${SOFAGENERAL_BUILD_CMAKE_DIR}/SofaGeneralConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAGENERAL_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAGENERAL_INCLUDE_DIR )

install(FILES ${SOFAGENERAL_BUILD_CMAKE_DIR}/SofaGeneralConfigInstall.cmake
        DESTINATION ${SOFAGENERAL_INSTALL_CMAKE_DIR}
        RENAME SofaGeneralConfig.cmake)

# SofaGeneralTargets.cmake (installed)
install(EXPORT SofaGeneralTargets
        DESTINATION ${SOFAGENERAL_INSTALL_CMAKE_DIR} )
