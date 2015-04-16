cmake_minimum_required(VERSION 2.8.12)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include(CMakeSofaCreateTarget)

set(SOLUTION_NAME "SofaFramework")
project(${SOLUTION_NAME})

# Temporary variable until we move the sub projects
set( SofaFramework_ACTUAL_SOURCE_DIR ${SofaFramework_SOURCE_DIR}/../framework)

##
## SofaFramework options
##
option(SOFAFRAMEWORK_DETECTIONOUTPUT_FREEMOTION      "Compile SofaCore library with DETECTIONOUTPUT_FREEMOTION preprocessor definition" OFF)
option(SOFAFRAMEWORK_DETECTIONOUTPUT_BARYCENTRICINFO "Compile SofaCore library with DETECTIONOUTPUT_BARYCENTRICINFO preprocessor definition" OFF)
option(SOFA_NO_OPENGL                                "Compile SofaFramework libraries without OpenGL dependencies" OFF )

set(SOFA_FLOATING_POINT_TYPE double CACHE STRING
"Type used for floating point values in SOFA.  It actually determines:
 - what template instanciations will be compiled (via the definition of the
   SOFA_FLOAT and SOFA_DOUBLE macros)
 - what is the type behind the 'SReal' typedef used throughout SOFA. (If 'both'
   is selected, SReal defaults to double.)" )
set_property(CACHE SOFA_FLOATING_POINT_TYPE PROPERTY STRINGS float double both)

if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_DOUBLE 1)
elseif(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_FLOAT 1)
endif()

set(DETECTIONOUTPUT_FREEMOTION ${SOFAFRAMEWORK_DETECTIONOUTPUT_FREEMOTION} )
set(DETECTIONOUTPUT_BARYCENTRICINFO ${SOFAFRAMEWORK_DETECTIONOUTPUT_BARYCENTRICINFO} )


set( SOFAFRAMEWORK_EXTLIBS_DIR "${SofaFramework_ACTUAL_SOURCE_DIR}/../extlibs")

set(SOFAFRAMEWORK_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFAFRAMEWORK_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFAFRAMEWORK_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAFRAMEWORK_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaFramework)

# Offer the user the choice of overriding the installation directories
set(SOFAFRAMEWORK_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAFRAMEWORK_INSTALL_CMAKE_DIR   ${SOFAFRAMEWORK_INSTALL_LIB_DIR}/cmake/SofaFramework  )

set(SOFAFRAMEWORK_DEPENDENCY_LINK)
set(SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES)

##
## Dependencies lookup 
##

find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
    set(SOFA_HAVE_GTEST ${GTest_FOUND})
else()
    set(SOFA_HAVE_GTEST ${GTest_FOUND})
endif()

if( NOT SOFA_NO_OPENGL )

    # OpenGL ( glu )
    find_package(OpenGL REQUIRED)
    # no target support for OpenGL on windows yet see https://github.com/Kitware/CMake/commit/02e34de2b6c27ee5d367be4fa280ffd3ad1965ba
    # it is not too harmful since this particular library gives symbolic names that can be consumed upon install on the target system.
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${OPENGL_LIBRARIES})
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${OPENGL_INCLUDE_DIR})

    # GLEW
    find_package(GLEW)
    if( GLEW_FOUND)
        sofa_create_target(GLEW SofaFramework "${GLEW_LIBRARIES}" "${GLEW_INCLUDE_DIRS}" )
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${GLEW_Target} )
    endif()
    set(SOFA_HAVE_GLEW ${GLEW_FOUND})

    # GLUT
    find_package(GLUT)
    if(GLUT_FOUND)
        if(NOT APPLE)
            sofa_create_target(GLUT SofaFramework "${GLUT_glut_LIBRARY}" "${GLUT_INCLUDE_DIR}" )
            list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${GLUT_Target})
        else()
            # Workaround: on OS X, "${GLUT_glut_LIBRARY}" is not a target (it is
            # actually "-framework GLUT"), so we use find_library to find
            # something better (like /System/Library/Frameworks/GLUT.framework.)
            # (Fun fact: FindGLUT.cmake litterally reads "These values for Apple
            # could probably do with improvement.)"
            find_library(ACTUAL_GLUT_LIBRARIES "GLUT")
            list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK "${ACTUAL_GLUT_LIBRARIES}")
        endif()
    endif()
    set(SOFA_HAVE_GLUT ${GLUT_FOUND})
    
endif() 

#tinyxml
find_package(TinyXML REQUIRED)
sofa_create_target(TinyXML SofaFramework "${TinyXML_LIBRARIES}" "${TinyXML_INCLUDE_DIRS}" )
list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${TinyXML_Target})
set(SOFA_HAVE_TINYXML 1)

#zlib
find_package(ZLIB)
if(ZLIB_FOUND)
    sofa_create_target(ZLIB SofaFramework "${ZLIB_LIBRARIES}" "${ZLIB_INCLUDE_DIRS}" )
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${ZLIB_Target})
endif()
set(SOFA_HAVE_ZLIB ${ZLIB_FOUND})

# PNG
find_package(PNG)
if( PNG_FOUND)
    sofa_create_target(PNG SofaFramework "${PNG_LIBRARY}" "${PNG_INCLUDE_DIRS}" )
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${PNG_Target} )
endif()
set(SOFA_HAVE_PNG ${PNG_FOUND})

# Boost
find_package(Boost COMPONENTS thread system date_time chrono QUIET)

set(SOFA_HAVE_BOOST_SYSTEM ${Boost_SYSTEM_FOUND} )
set(SOFA_HAVE_BOOST_THREAD ${Boost_THREAD_FOUND} )
set(SOFA_HAVE_BOOST_DATE_TIME ${Boost_DATE_TIME_FOUND} )
set(SOFA_HAVE_BOOST_CHRONO ${Boost_CHRONO_FOUND})

list(APPEND Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR} )

if(SOFA_HAVE_BOOST_SYSTEM AND 
   SOFA_HAVE_BOOST_THREAD AND 
   SOFA_HAVE_BOOST_DATE_TIME AND 
   SOFA_HAVE_BOOST_CHRONO)
    set(SOFA_HAVE_BOOST 1)
    
    sofa_create_target(BoostSystem SofaFramework "${Boost_SYSTEM_LIBRARY}" "${Boost_INCLUDE_DIRS}" )
    sofa_create_target(BoostThread SofaFramework "${Boost_THREAD_LIBRARY}" "${Boost_INCLUDE_DIRS}" )
    sofa_create_target(BoostDateTime SofaFramework "${Boost_DATE_TIME_LIBRARY}" "${Boost_INCLUDE_DIRS}" )
    sofa_create_target(BoostChrono SofaFramework "${Boost_CHRONO_LIBRARY}" "${Boost_INCLUDE_DIRS}" )
    
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostSystem_Target} )
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostThread_Target} )
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostDateTime_Target} )
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostChrono_Target} )
else()
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif()

#Eigen
find_package(Eigen3 REQUIRED)
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
set(SOFA_HAVE_EIGEN2 1)
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
if(SOFA_HAVE_GTEST)
    add_subdirectory( ${SofaFramework_ACTUAL_SOURCE_DIR}/framework_test        ${CMAKE_CURRENT_BINARY_DIR}/framework_test )
endif()

# IDE folder
set_target_properties(SofaHelper PROPERTIES FOLDER "SofaFramework")
set_target_properties(SofaDefaultType PROPERTIES FOLDER "SofaFramework")
set_target_properties(SofaCore PROPERTIES FOLDER "SofaFramework")


include(CMakePackageConfigHelpers)

## Version

# Not really pretty, just a number that satisfies SOFA_VERSION > LAST_RELEASED_VERSION
set(SOFA_VERSION "999999")
set(SOFA_VERSION_STR "\"devel\"")
set(SOFAFRAMEWORK_VERSION "99.99.99")
# Generate sofa/version.h
configure_file(${SofaFramework_ACTUAL_SOURCE_DIR}/sofa/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/sofa/version.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/version.h
        DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}/sofa")

# SofaFrameworkConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigVersion.cmake
                                 VERSION ${SOFAFRAMEWORK_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigVersion.cmake
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR})


# SofaFrameworkConfig.cmake (build tree)
set(SOFAFRAMEWORK_INCLUDE_DIRS "${SOFAFRAMEWORK_BUILD_INCLUDE_DIR}" "${SofaFramework_ACTUAL_SOURCE_DIR}")
configure_package_config_file(SofaFrameworkConfig.cmake.in
                              ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfig.cmake
                              INSTALL_DESTINATION ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}
                              PATH_VARS SOFAFRAMEWORK_INCLUDE_DIRS)

# SofaFrameworkConfig.cmake (installed)
set(SOFAFRAMEWORK_INCLUDE_DIRS "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaFrameworkConfig.cmake.in
                              ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAFRAMEWORK_INCLUDE_DIRS)

install(FILES ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/SofaFrameworkConfigInstall.cmake
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR}
        RENAME SofaFrameworkConfig.cmake)

# CMakeSofaCreateTarget (build)
configure_file( ${SofaFramework_SOURCE_DIR}/CMakeSofaCreateTarget.cmake
                ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/CMakeSofaCreateTarget.cmake COPYONLY)
        
# CMakeSofaCreateTarget (installed)
install(FILES ${SofaFramework_SOURCE_DIR}/CMakeSofaCreateTarget.cmake 
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR} )

# CMakeParseLibraryList (build)
configure_file( ${SofaFramework_SOURCE_DIR}/CMakeParseLibraryList.cmake
                ${SOFAFRAMEWORK_BUILD_CMAKE_DIR}/CMakeParseLibraryList.cmake COPYONLY)
        
# CMakeParseLibraryList (installed)
install(FILES ${SofaFramework_SOURCE_DIR}/CMakeParseLibraryList.cmake 
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR} )
        
# SofaFrameworkTargets.cmake (installed)
install(EXPORT SofaFrameworkTargets
        DESTINATION ${SOFAFRAMEWORK_INSTALL_CMAKE_DIR} )
