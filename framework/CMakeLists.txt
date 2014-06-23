cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaFramework")
project(${SOLUTION_NAME})
message("CMAKE_FILES_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}" )
message("Install prefix: ${CMAKE_INSTALL_PREFIX}")

##
## SofaFramework version 
##
set( SOFAFRAMEWORK_MAJOR_VERSION 0 )
set( SOFAFRAMEWORK_MINOR_VERSION 0 )
set( SOFAFRAMEWORK_BUILD_VERSION 1 )

set( SOFAFRAMEWORK_VERSION ${SOFAFRAMEWORK_MAJOR_VERSION}.${SOFAFRAMEWORK_MINOR_VERSION}.${SOFAFRAMEWORK_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.") 

# MSVC: Use solution folders.
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()


set( SOFAFRAMEWORK_EXTLIBS_DIR "${SofaFramework_SOURCE_DIR}/../extlibs")
set( SOFAFRAMEWORK_INCLUDE_DIR "${SofaFramework_SOURCE_DIR}")

set(SOFAFRAMEWORK_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAFRAMEWORK_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAFRAMEWORK_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SOFAFRAMEWORK_BUILD_CMAKE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/SofaFramework)

message("Build dir : ${CMAKE_CURRENT_BINARY_DIR}")

# Offer the user the choice of overriding the installation directories
set(SOFAFRAMEWORK_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_CMAKE_DIR   lib/cmake/SofaFramework  CACHE PATH "Installation directory for CMake files relative to CMAKE_INSTALL_PREFIX variable")

if( CMAKE_CL_64 )
  set( LIB "lib/win64" )
else( CMAKE_CL_64 )
  set( LIB "lib/win32" )
endif( CMAKE_CL_64 )

##
## System Dependencies lookup 
##
set(SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS)

# zlib
if(WIN32)
    set(SOFAFRAMEWORK_USE_IMPORTED_ZLIB TRUE CACHE BOOL "")
    set(SOFAFRAMEWORK_ZLIB_IMPORTED_LOCATION "${SofaFramework_SOURCE_DIR}/../${LIB}/zlib1.dll" CACHE PATH "" )
    set(SOFAFRAMEWORK_ZLIB_IMPORTED_IMPLIB   "${SofaFramework_SOURCE_DIR}/../${LIB}/zlib.lib" CACHE PATH "" )
    set(SOFAFRAMEWORK_ZLIB_INCLUDE_DIRS      "${SofaFramework_SOURCE_DIR}/../include" CACHE PATH ""         )
    
    set(SOFAFRAMEWORK_USE_IMPORTED_PNG TRUE CACHE BOOL "" )
    set(SOFAFRAMEWORK_PNG_IMPORTED_LOCATION "${SofaFramework_SOURCE_DIR}/../${LIB}/libpng15.dll" CACHE PATH "" )
    set(SOFAFRAMEWORK_PNG_IMPORTED_IMPLIB   "${SofaFramework_SOURCE_DIR}/../${LIB}/libpng.lib" CACHE PATH ""   )
    set(SOFAFRAMEWORK_PNG_INCLUDE_DIRS      "${SofaFramework_SOURCE_DIR}/../include" CACHE PATH ""             )
    
    set(SOFAFRAMEWORK_USE_IMPORTED_GLEW TRUE CACHE BOOL "" )
    set(SOFAFRAMEWORK_GLEW_IMPORTED_LOCATION "${SofaFramework_SOURCE_DIR}/../${LIB}/glew32.dll" CACHE PATH ""   )
    set(SOFAFRAMEWORK_GLEW_IMPORTED_IMPLIB   "${SofaFramework_SOURCE_DIR}/../${LIB}/glew32.lib" CACHE PATH ""   )
    set(SOFAFRAMEWORK_GLEW_INCLUDE_DIRS      "${SofaFramework_SOURCE_DIR}/../include" CACHE PATH ""             )
    
    set(SOFAFRAMEWORK_USE_IMPORTED_GLUT TRUE CACHE BOOL "" )
    if(CMAKE_CL_64)
        set(SOFAFRAMEWORK_GLUT_IMPORTED_LOCATION "${SofaFramework_SOURCE_DIR}/../${LIB}/glut64.dll" CACHE PATH "" FORCE)
    else( CMAKE_CL_64 )
        set(SOFAFRAMEWORK_GLUT_IMPORTED_LOCATION "${SofaFramework_SOURCE_DIR}/../${LIB}/glut32.dll" CACHE PATH "" FORCE)
    endif( CMAKE_CL_64 )
    set(SOFAFRAMEWORK_GLUT_IMPORTED_IMPLIB   "${SofaFramework_SOURCE_DIR}/../${LIB}/glut32.lib" CACHE PATH ""   )
    set(SOFAFRAMEWORK_GLUT_INCLUDE_DIRS      "${SofaFramework_SOURCE_DIR}/../include" CACHE PATH ""             )
endif()

# Zlib
if( SOFAFRAMEWORK_USE_IMPORTED_ZLIB )
    add_library(zlib SHARED IMPORTED )
    set_property(TARGET zlib PROPERTY IMPORTED_LOCATION ${SOFAFRAMEWORK_ZLIB_IMPORTED_LOCATION}) 
    set_property(TARGET zlib PROPERTY IMPORTED_IMPLIB   ${SOFAFRAMEWORK_ZLIB_IMPORTED_IMPLIB}  )
    include_directories(${SOFAFRAMEWORK_ZLIB_INCLUDE_DIRS})
    set(ZLIB_INCLUDE_DIR ${SOFAFRAMEWORK_ZLIB_INCLUDE_DIRS} )
	set(ZLIB_LIBRARIES zlib)
	
    install(DIRECTORY ${SOFAFRAMEWORK_ZLIB_INCLUDE_DIRS}/ 
        COMPONENT SofaFramework_headers
        DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
    )
    install(FILES ${SOFAFRAMEWORK_ZLIB_IMPORTED_LOCATION}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_BIN_DIR}"
    )
    install(FILES ${SOFAFRAMEWORK_ZLIB_IMPORTED_IMPLIB}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_LIB_DIR}"
    )
else()
    find_package(ZLIB REQUIRED)
    include_directories(${ZLIB_INCLUDE_DIR})
endif()
list(APPEND SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR})
set(SOFA_HAVE_ZLIB 1)

# PNG
if( SOFAFRAMEWORK_USE_IMPORTED_PNG )
    add_library(png SHARED IMPORTED )
    set_property(TARGET png PROPERTY IMPORTED_LOCATION ${SOFAFRAMEWORK_PNG_IMPORTED_LOCATION}) 
    set_property(TARGET png PROPERTY IMPORTED_IMPLIB   ${SOFAFRAMEWORK_PNG_IMPORTED_IMPLIB}  )
    include_directories(${SOFAFRAMEWORK_PNG_INCLUDE_DIRS} )
    set(PNG_INCLUDE_DIRS ${SOFAFRAMEWORK_PNG_INCLUDE_DIRS} )
	set(PNG_LIBRARIES png)
    install(DIRECTORY ${SOFAFRAMEWORK_PNG_INCLUDE_DIRS}/ 
        COMPONENT SofaFramework_headers
        DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
    )
    install(FILES ${SOFAFRAMEWORK_PNG_IMPORTED_LOCATION}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_BIN_DIR}"
    )
    install(FILES ${SOFAFRAMEWORK_PNG_IMPORTED_IMPLIB}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_LIB_DIR}"
    )
else() 
    find_package(PNG  REQUIRED)
    include_directories(${PNG_INCLUDE_DIRS})
endif()
list(APPEND SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS ${PNG_INCLUDE_DIRS})
set(SOFA_HAVE_PNG 1)

# GLEW 
if( SOFAFRAMEWORK_USE_IMPORTED_GLEW )
    add_library(glew SHARED IMPORTED )
    set_property(TARGET glew PROPERTY IMPORTED_LOCATION ${SOFAFRAMEWORK_GLEW_IMPORTED_LOCATION}) 
    set_property(TARGET glew PROPERTY IMPORTED_IMPLIB   ${SOFAFRAMEWORK_GLEW_IMPORTED_IMPLIB}  )
    include_directories(${SOFAFRAMEWORK_GLEW_INCLUDE_DIRS} )
    set(GLEW_INCLUDE_DIRS ${SOFAFRAMEWORK_GLEW_INCLUDE_DIRS} )
	set(GLEW_LIBRARIES glew)
    install(DIRECTORY ${SOFAFRAMEWORK_GLEW_INCLUDE_DIRS}/ 
        COMPONENT SofaFramework_headers
        DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
    )
    install(FILES ${SOFAFRAMEWORK_GLEW_IMPORTED_LOCATION}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_BIN_DIR}"
    )
    install(FILES ${SOFAFRAMEWORK_GLEW_IMPORTED_IMPLIB}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_LIB_DIR}"
    )
else() 
    find_package(GLEW REQUIRED)
    set(SOFA_HAVE_GLEW 1)
    include_directories(${GLEW_INCLUDE_DIRS})
endif()
list(APPEND SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS})
set(SOFA_HAVE_GLEW 1)

# GLUT 
if( SOFAFRAMEWORK_USE_IMPORTED_GLUT )
    add_library(glut SHARED IMPORTED )
    set_property(TARGET glut PROPERTY IMPORTED_LOCATION ${SOFAFRAMEWORK_GLUT_IMPORTED_LOCATION}) 
    set_property(TARGET glut PROPERTY IMPORTED_IMPLIB   ${SOFAFRAMEWORK_GLUT_IMPORTED_IMPLIB}  )
    include_directories(${SOFAFRAMEWORK_GLUT_INCLUDE_DIRS} )
    set(GLUT_INCLUDE_DIR ${SOFAFRAMEWORK_GLUT_INCLUDE_DIRS} )
	set(GLUT_LIBRARIES glut )
    install(DIRECTORY ${SOFAFRAMEWORK_GLUT_INCLUDE_DIRS}/ 
        COMPONENT SofaFramework_headers
        DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}"
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
    )
    install(FILES ${SOFAFRAMEWORK_GLUT_IMPORTED_LOCATION}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_BIN_DIR}"
    )
    install(FILES ${SOFAFRAMEWORK_GLUT_IMPORTED_IMPLIB}
        COMPONENT SofaFramework_libraries
        DESTINATION "${SOFAFRAMEWORK_INSTALL_LIB_DIR}"
    )
else() 
    find_package(GLUT REQUIRED)
    include_directories(${GLUT_INCLUDE_DIR})
endif()
list(APPEND SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS ${GLUT_INCLUDE_DIR})
set(SOFA_HAVE_GLUT 1)

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
else()
    find_package(OpenGL REQUIRED)
    include_directories(${OPENGL_INCLUDE_DIR})
endif(PS3)
list(APPEND SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})
# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(SOFA_USE_MINIBOOST 0)
set(Boost_FIND_QUIETLY 1)
find_package(Boost COMPONENTS thread system)
if(Boost_FOUND)
    if(Boost_thread_FOUND AND Boost_system_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
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
    include_directories(${Boost_INCLUDE_DIRS})
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
list(APPEND SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
##
##
##
list(REMOVE_DUPLICATES SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS)

##
## Build/Install tree dependencies 
##
set(SOFAFRAMEWORK_TARGETS)
#tinyxml
find_package(TinyXML  REQUIRED)
if( TinyXML_TARGETS )  # if defined TinyXML belongs to current build tree
	list(APPEND SOFAFRAMEWORK_TARGETS tinyxml ) # if so add TinyXML to the export target list for transitive dependency management within the build tree.
endif()
set(SOFA_HAVE_TINYXML 1)
##
##
##

configure_file( ${SofaFramework_SOURCE_DIR}/SofaFramework.config.in 
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaFramework.h )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaFramework.h 
         DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}/sofa")
# make sure everyone in the build tree can see <sofa/SofaFramework.h>
include_directories(${CMAKE_CURRENT_BINARY_DIR})


list(APPEND SOFAFRAMEWORK_TARGETS SofaCore SofaHelper SofaDefaultType)
set(SOFAFRAMEWORK_EXPORT_FILE SofaFrameworkTargets.cmake)

add_subdirectory( ${SofaFramework_SOURCE_DIR}/sofa/helper      ${CMAKE_CURRENT_BINARY_DIR}/SofaHelper )
add_subdirectory( ${SofaFramework_SOURCE_DIR}/sofa/defaulttype ${CMAKE_CURRENT_BINARY_DIR}/SofaDefaultType )
add_subdirectory( ${SofaFramework_SOURCE_DIR}/sofa/core        ${CMAKE_CURRENT_BINARY_DIR}/SofaCore )

 
include(CMakePackageConfigHelpers)

# SofaFrameworkConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigVersion.cmake
                                 VERSION ${SOFAFRAMEWORK_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigVersion.cmake
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR})
                                                          

# SofaFrameworkConfig.cmake (build tree)
set(SOFAFRAMEWORK_INCLUDE_DIR "${SOFAFRAMEWORK_BUILD_INCLUDE_DIR}")
set(SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIR "${SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIRS}")
set(SOFAFRAMEWORK_BIN_DIR "${SOFAFRAMEWORK_BUILD_BIN_DIR}")
configure_package_config_file(SofaFrameworkConfig.cmake.in 
                              ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfig.cmake
                              INSTALL_DESTINATION ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}
                              PATH_VARS SOFAFRAMEWORK_INCLUDE_DIR
                                        SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIR
                                        SOFAFRAMEWORK_BIN_DIR)
                          
# SofaFrameworkConfig.cmake (installed)
set(SOFAFRAMEWORK_INCLUDE_DIR "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}")
set(SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIR "")
set(SOFAFRAMEWORK_BIN_DIR "")
configure_package_config_file(SofaFrameworkConfig.cmake.in 
                              ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAFRAMEWORK_INCLUDE_DIR
                                        SOFAFRAMEWORK_EXTERNAL_INCLUDE_DIR
                                        SOFAFRAMEWORK_BIN_DIR)
                          
install(FILES ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigInstall.cmake
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR}
        RENAME SofaFrameworkConfig.cmake)

# SofaFrameworkTargets.cmake (build tree)        
export(TARGETS ${SOFAFRAMEWORK_TARGETS}
       FILE ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkTargets.cmake )                                        
                                                          
# SofaFrameworkTargets.cmake (installed)
install(EXPORT SofaFrameworkTargets 
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR} )

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE SofaFramework)

#include(SofaFrameworkCPack.cmake)

# Tests
add_subdirectory(framework_test)
