cmake_minimum_required(VERSION 2.8.12)

project(Eigen)

##
## Eigen version
##
set( EIGEN_MAJOR_VERSION 3 )
set( EIGEN_MINOR_VERSION 2 )
set( EIGEN_BUILD_VERSION 1 )

set(EIGEN_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(EIGEN_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/Eigen)

set(EIGEN_INSTALL_INCLUDE_DIR include/sofa    CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(EIGEN_INSTALL_LIB_DIR     lib              CACHE PATH "Installation directory for CMake files relative to CMAKE_INSTALL_PREFIX variable")
set(EIGEN_INSTALL_CMAKE_DIR   lib/cmake/Eigen  CACHE PATH "Installation directory for CMake files relative to CMAKE_INSTALL_PREFIX variable")

set(EIGEN_VERSION ${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}.${EIGEN_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

# only take the librairy headers, not test headers and other stuff
file(GLOB_RECURSE Eigen_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Eigen/*.h ${CMAKE_CURRENT_SOURCE_DIR}/unsupported/Eigen/*.h)

add_library(SofaEigenHeaders STATIC ${Eigen_HEADERS}  )
target_include_directories(SofaEigenHeaders PUBLIC "$<BUILD_INTERFACE:${EIGEN_BUILD_INCLUDE_DIR}>") 
target_include_directories(SofaEigenHeaders PUBLIC "$<INSTALL_INTERFACE:${EIGEN_INSTALL_INCLUDE_DIR}>") 
set_target_properties(SofaEigenHeaders PROPERTIES LINKER_LANGUAGE CXX)

include(CMakePackageConfigHelpers)

# EigenConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${EIGEN_BUILD_CMAKE_DIR}/Eigen3ConfigVersion.cmake
                                 VERSION ${EIGEN_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${EIGEN_BUILD_CMAKE_DIR}/Eigen3ConfigVersion.cmake
        DESTINATION ${EIGEN_INSTALL_CMAKE_DIR})

# EigenConfig.cmake (build tree)
set(EIGEN_INCLUDE_DIR "${EIGEN_BUILD_INCLUDE_DIR}")
configure_package_config_file(EigenConfig.cmake.in
                              ${EIGEN_BUILD_CMAKE_DIR}/Eigen3Config.cmake
                              INSTALL_DESTINATION ${EIGEN_BUILD_CMAKE_DIR}
                              PATH_VARS EIGEN_INCLUDE_DIR )

# EigenConfig.cmake (installed)
set(EIGEN_INCLUDE_DIR "${EIGEN_INSTALL_INCLUDE_DIR}")
configure_package_config_file(Eigen3Config.cmake.in
                              ${EIGEN_BUILD_CMAKE_DIR}/Eigen3ConfigInstall.cmake
                              INSTALL_DESTINATION ${EIGEN_INSTALL_CMAKE_DIR}
                              PATH_VARS EIGEN_INCLUDE_DIR )

install(FILES ${EIGEN_BUILD_CMAKE_DIR}/EigenConfigInstall.cmake
        DESTINATION ${EIGEN_INSTALL_CMAKE_DIR}
        RENAME Eigen3Config.cmake)

install(TARGETS SofaEigenHeaders
        EXPORT EigenTargets
        ARCHIVE DESTINATION "${EIGEN_INSTALL_LIB_DIR}")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ 
        DESTINATION "${EIGEN_INSTALL_INCLUDE_DIR}"
        COMPONENT SofaEigen_headers
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )

# EigenTargets.cmake (installed)
install(EXPORT EigenTargets
        DESTINATION ${EIGEN_INSTALL_CMAKE_DIR} )
