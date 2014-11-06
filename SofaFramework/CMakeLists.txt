cmake_minimum_required(VERSION 2.8.12)

set(SOLUTION_NAME "SofaFramework")
project(${SOLUTION_NAME})

# Temporary variable until we move the sub projects
set( SofaFramework_ACTUAL_SOURCE_DIR ${SofaFramework_SOURCE_DIR}/../framework)

##
## SofaFramework version
##
set( SOFAFRAMEWORK_MAJOR_VERSION 0 )
set( SOFAFRAMEWORK_MINOR_VERSION 0 )
set( SOFAFRAMEWORK_BUILD_VERSION 1 )

set( SOFAFRAMEWORK_VERSION ${SOFAFRAMEWORK_MAJOR_VERSION}.${SOFAFRAMEWORK_MINOR_VERSION}.${SOFAFRAMEWORK_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set( SOFAFRAMEWORK_EXTLIBS_DIR "${SofaFramework_ACTUAL_SOURCE_DIR}/../extlibs")

set(SOFAFRAMEWORK_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFAFRAMEWORK_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFAFRAMEWORK_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAFRAMEWORK_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaFramework)

# Offer the user the choice of overriding the installation directories
set(SOFAFRAMEWORK_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_CMAKE_DIR   lib/cmake/SofaFramework  CACHE PATH "Installation directory for CMake files relative to CMAKE_INSTALL_PREFIX variable")

set(SOFAFRAMEWORK_DEPENDENCY_LINK)
set(SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES)
##
## System Dependencies lookup (mandatory)
##
# OpenGL ( glu )
if(PS3)
    set(OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/PSGL/RSX/ultra-opt/libPSGL.a")
    list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/PSGL/RSX/ultra-opt/libPSGLU.a")
    list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libgcm_cmd.a")
    list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libgcm_pm.a")
    list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libgcm_sys_stub.a")
    list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libsysutil_avconf_ext_stub.a")
    list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libresc_stub.a")
    list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libsysutil_stub.a")
    message( "OPENGL PS3 LIBS" ${OPENGL_LIBRARIES} )
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${OPENGL_LIBRARIES} )
else()
    find_package(OpenGL REQUIRED)
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${OPENGL_INCLUDE_DIR})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${OPENGL_LIBRARIES} )
endif(PS3)
##
##
##

##
## Sofa Extlibs dependencies lookup (mandatory)
##
#tinyxml
find_package(TinyXML  REQUIRED)
list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${TinyXML_LIBRARIES} )
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${TinyXML_INCLUDE_DIRS} )
set(SOFA_HAVE_TINYXML 1)

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(SOFA_USE_MINIBOOST 0)
set(Boost_FIND_QUIETLY 1)
find_package(Boost COMPONENTS thread system)
if(Boost_FOUND)
    if(Boost_thread_FOUND AND Boost_system_FOUND)
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
		list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
		list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${Boost_LIBRARIES} )
        set(SOFA_HAVE_BOOST        1)
        set(SOFA_HAVE_BOOST_SYSTEM 1)
        set(SOFA_HAVE_BOOST_THREAD 1)
    else(Boost_thread_FOUND AND Boost_system_FOUND)
        set(SOFA_USE_MINIBOOST 1)
    endif(Boost_thread_FOUND AND Boost_system_FOUND)
else(Boost_FOUND)
    set(SOFA_USE_MINIBOOST 1)
endif(Boost_FOUND)

if(SOFA_USE_MINIBOOST)
    message(STATUS "No boost install directory found use ${SOFAFRAMEWORK_EXTLIBS_DIR}/miniBoost instead")
    set(BOOST_ROOT "${SOFAFRAMEWORK_EXTLIBS_DIR}/miniBoost")
    find_package(Boost REQUIRED)
    #list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
	install(DIRECTORY ${Boost_INCLUDE_DIRS}/
            COMPONENT SofaFramework_headers
            DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}"
            FILES_MATCHING
                PATTERN "*.h"
                PATTERN "*.inl"
                PATTERN "*.hxx"
                PATTERN "*.hpp"
           )
endif(SOFA_USE_MINIBOOST)

#Eigen
set(Eigen_INCLUDE_DIRS "${SOFAFRAMEWORK_EXTLIBS_DIR}/eigen-3.2.1")
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Eigen_INCLUDE_DIRS})
install(DIRECTORY ${Eigen_INCLUDE_DIRS}/
        COMPONENT SofaFramework_headers
        DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
       )
set(SOFA_HAVE_EIGEN2 1)
##
##
##

##
## Optional Dependencies lookup
##
set(SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS)
#zlib
find_package(ZLIB)
if(ZLIB_FOUND)
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIRS})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${ZLIB_LIBRARIES})
endif()
set(SOFA_HAVE_ZLIB ${ZLIB_FOUND})
# PNG
find_package(PNG)
if( PNG_FOUND)
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${PNG_INCLUDE_DIRS})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${PNG_LIBRARIES})
endif()
set(SOFA_HAVE_PNG ${PNG_FOUND})
# GLEW
find_package(GLEW)
if( GLEW_FOUND)
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${GLEW_LIBRARIES})
endif()
set(SOFA_HAVE_GLEW ${GLEW_FOUND})
# GLUT
find_package(GLUT)
if( GLUT_FOUND )
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${GLUT_INCLUDE_DIR})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${GLUT_LIBRARIES})
endif()
set(SOFA_HAVE_GLUT ${GLUT_FOUND})
##
##
##

configure_file( ${SofaFramework_SOURCE_DIR}/SofaFramework.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaFramework.h )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaFramework.h
         DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}/sofa")
# make sure everyone in the build tree can see <sofa/SofaFramework.h>
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})

list(REMOVE_DUPLICATES SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES)

add_subdirectory( ${SofaFramework_ACTUAL_SOURCE_DIR}/sofa/helper      ${CMAKE_CURRENT_BINARY_DIR}/SofaHelper )
add_subdirectory( ${SofaFramework_ACTUAL_SOURCE_DIR}/sofa/defaulttype ${CMAKE_CURRENT_BINARY_DIR}/SofaDefaultType )
add_subdirectory( ${SofaFramework_ACTUAL_SOURCE_DIR}/sofa/core        ${CMAKE_CURRENT_BINARY_DIR}/SofaCore )
add_subdirectory( ${SofaFramework_ACTUAL_SOURCE_DIR}/framework_test        ${CMAKE_CURRENT_BINARY_DIR}/framework_test )

include(CMakePackageConfigHelpers)

# SofaFrameworkConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigVersion.cmake
                                 VERSION ${SOFAFRAMEWORK_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigVersion.cmake
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR})


# SofaFrameworkConfig.cmake (build tree)
set(SOFAFRAMEWORK_INCLUDE_DIR "${SOFAFRAMEWORK_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaFrameworkConfig.cmake.in
                              ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfig.cmake
                              INSTALL_DESTINATION ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}
                              PATH_VARS SOFAFRAMEWORK_INCLUDE_DIR)

# SofaFrameworkConfig.cmake (installed)
set(SOFAFRAMEWORK_INCLUDE_DIR "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaFrameworkConfig.cmake.in
                              ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAFRAMEWORK_INCLUDE_DIR)

install(FILES ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigInstall.cmake
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR}
        RENAME SofaFrameworkConfig.cmake)

# SofaFrameworkTargets.cmake (installed)
install(EXPORT SofaFrameworkTargets
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR} )
