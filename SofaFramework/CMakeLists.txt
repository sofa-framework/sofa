cmake_minimum_required(VERSION 2.8.12)
project(SofaFramework)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include(SofaMacros)


### Options

option(SOFA_DETECTIONOUTPUT_FREEMOTION
"Compile Sofa with the DETECTIONOUTPUT_FREEMOTION macro defined." OFF)
set(DETECTIONOUTPUT_FREEMOTION ${SOFA_DETECTIONOUTPUT_FREEMOTION}) # Used in sofa/config.h.in

option(SOFA_DETECTIONOUTPUT_BARYCENTRICINFO
"Compile Sofa with the DETECTIONOUTPUT_BARYCENTRICINFO macro defined." OFF)
set(DETECTIONOUTPUT_BARYCENTRICINFO ${SOFA_DETECTIONOUTPUT_BARYCENTRICINFO}) # Used in sofa/config.h.in

option(SOFA_NO_OPENGL
"Compile Sofa with no OpenGL support. (This will define the SOFA_NO_OPENGL
macro.)" OFF)

option(SOFA_NO_EXTERN_TEMPLATE
"Compile Sofa without using the extern templates mechanism. (This will prevent
the SOFA_EXTERN_TEMPLATE macro from being defined.)" OFF)

option(SOFA_NO_UPDATE_BBOX
"Compile Sofa with the SOFA_NO_UPDATE_BBOX macro defined." OFF)

option(SOFA_DUMP_VISITOR_INFO
"Compile Sofa with the SOFA_DUMP_VISITOR_INFO macro defined." OFF)

set(SOFA_FLOATING_POINT_TYPE both CACHE STRING
"Type used for floating point values in SOFA. It actually determines:
 - what template instanciations will be compiled (via the definition of the
   SOFA_FLOAT and SOFA_DOUBLE macros)
 - what is the type behind the 'SReal' typedef used throughout SOFA. (If 'both'
   is selected, SReal defaults to double.)")
set_property(CACHE SOFA_FLOATING_POINT_TYPE PROPERTY STRINGS float double both)


if(MSVC)
    option(SOFA_VECTORIZE "Enable the use of SSE2 instructions by the compiler (Only available for MSVC)." OFF)
endif()

if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_DOUBLE 1)          # Used in sofa/config.h.in
elseif(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_FLOAT 1)           # Used in sofa/config.h.in
endif()



### Dependencies

set(SOFAFRAMEWORK_DEPENDENCY_LINK)
set(SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES)


find_package(GTest CONFIG QUIET)
set(SOFA_HAVE_GTEST ${GTest_FOUND})

if(NOT SOFA_NO_OPENGL)

    # OpenGL (glu)
    find_package(OpenGL REQUIRED)
    # no target support for OpenGL on windows yet see
    # https://github.com/Kitware/CMake/commit/02e34de2b6c27ee5d367be4fa280ffd3ad1965ba
    # it is not too harmful since this particular library gives symbolic names
    # that can be consumed upon install on the target system.
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${OPENGL_LIBRARIES})
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${OPENGL_INCLUDE_DIR})

    ## GLEW
    find_package(GLEW)
    if(GLEW_FOUND)
        sofa_create_target(GLEW SofaFramework "${GLEW_LIBRARIES}" "${GLEW_INCLUDE_DIRS}")
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${GLEW_Target})
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})

        sofa_install_get_libraries(${GLEW_LIBRARIES})
    else()
        message("OpenGL advanced functions (e.g shaders, FBO) are disabled.")
    endif()
    set(SOFA_HAVE_GLEW ${GLEW_FOUND})

    ## GLUT
    find_package(GLUT)
    if(GLUT_FOUND)
        if(NOT APPLE)
            sofa_create_target(GLUT SofaFramework "${GLUT_glut_LIBRARY}" "${GLUT_INCLUDE_DIR}")
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

## tinyxml
find_package(TinyXML REQUIRED)
sofa_create_target(TinyXML SofaFramework "${TinyXML_LIBRARIES}" "${TinyXML_INCLUDE_DIRS}")
list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${TinyXML_Target})
set(SOFA_HAVE_TINYXML 1)

## zlib
find_package(ZLIB)
if(ZLIB_FOUND)
    sofa_create_target(ZLIB SofaFramework "${ZLIB_LIBRARIES}" "${ZLIB_INCLUDE_DIRS}")
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${ZLIB_Target})

    sofa_install_get_libraries(${ZLIB_LIBRARIES})
endif()
set(SOFA_HAVE_ZLIB ${ZLIB_FOUND})

## PNG
find_package(PNG)
if(PNG_FOUND)
    sofa_create_target(PNG SofaFramework "${PNG_LIBRARY}" "${PNG_INCLUDE_DIRS}")
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${PNG_Target})


    sofa_install_get_libraries("${PNG_LIBRARY}/libpng")
endif()
set(SOFA_HAVE_PNG ${PNG_FOUND})

## Boost
find_package(Boost COMPONENTS thread system date_time chrono QUIET)

set(SOFA_HAVE_BOOST_SYSTEM ${Boost_SYSTEM_FOUND})
set(SOFA_HAVE_BOOST_THREAD ${Boost_THREAD_FOUND})
set(SOFA_HAVE_BOOST_DATE_TIME ${Boost_DATE_TIME_FOUND})
set(SOFA_HAVE_BOOST_CHRONO ${Boost_CHRONO_FOUND})

list(APPEND Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})

if(SOFA_HAVE_BOOST_SYSTEM
   AND SOFA_HAVE_BOOST_THREAD
   AND SOFA_HAVE_BOOST_DATE_TIME
   AND SOFA_HAVE_BOOST_CHRONO)
    set(SOFA_HAVE_BOOST 1)

    sofa_create_target(BoostSystem SofaFramework "${Boost_SYSTEM_LIBRARY}" "${Boost_INCLUDE_DIRS}")
    sofa_create_target(BoostThread SofaFramework "${Boost_THREAD_LIBRARY}" "${Boost_INCLUDE_DIRS}")
    sofa_create_target(BoostDateTime SofaFramework "${Boost_DATE_TIME_LIBRARY}" "${Boost_INCLUDE_DIRS}")
    sofa_create_target(BoostChrono SofaFramework "${Boost_CHRONO_LIBRARY}" "${Boost_INCLUDE_DIRS}")

	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostSystem_Target})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostThread_Target})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostDateTime_Target})
	list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostChrono_Target})

    sofa_install_get_libraries(${Boost_SYSTEM_LIBRARY})
    sofa_install_get_libraries(${Boost_THREAD_LIBRARY})
    sofa_install_get_libraries(${Boost_DATE_TIME_LIBRARY})
    sofa_install_get_libraries(${Boost_CHRONO_LIBRARY})
else()
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif()

## Eigen
find_package(Eigen3 REQUIRED)
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})


## sofa/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/sofa/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config.h" DESTINATION "include/sofa")
# make sure everyone in the build tree can see <sofa/config.h>
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})
list(REMOVE_DUPLICATES SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES)

## Version
set(SOFAFRAMEWORK_VERSION "15.12.0")

## sofa/version.h
set(SOFA_VERSION "151200")
set(SOFA_VERSION_STR "\"devel\"")
configure_file(../framework/sofa/version.h.in "${CMAKE_BINARY_DIR}/include/sofa/version.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/version.h" DESTINATION include/sofa)


## Subdirectories
add_subdirectory(../framework/sofa/helper ${CMAKE_CURRENT_BINARY_DIR}/SofaHelper)
add_subdirectory(../framework/sofa/defaulttype ${CMAKE_CURRENT_BINARY_DIR}/SofaDefaultType)
add_subdirectory(../framework/sofa/core ${CMAKE_CURRENT_BINARY_DIR}/SofaCore)
if(SOFA_HAVE_GTEST)
    add_subdirectory(../framework/framework_test ${CMAKE_CURRENT_BINARY_DIR}/framework_test)
endif()

# IDE folder
set_target_properties(SofaHelper PROPERTIES FOLDER "SofaFramework")
set_target_properties(SofaDefaultType PROPERTIES FOLDER "SofaFramework")
set_target_properties(SofaCore PROPERTIES FOLDER "SofaFramework")


# NDEBUG preprocessor macro
if(WIN32 OR APPLE)
    # NDEBUG and _DEBUG are automatically set in the default c/cxx flags of the right configurations
elseif(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(SofaHelper PUBLIC "NDEBUG")
endif()

if(WIN32 OR XBOX)
    # Increase Number of Sections in .Obj file
    target_compile_options(SofaHelper PUBLIC "/bigobj")
endif()

# SSE2 flags
if(MSVC)
    if(SOFA_VECTORIZE)
        target_compile_options(SofaHelper PUBLIC "/arch:SSE2 /fp:fast")
    endif()
endif()

# C++11
if(NOT WIN32)
    if(SOFA_USE_C++11)

        # If we compile in c++11, then everyone linking against us should, so we
        # put -std=c++11 in the INTERFACE_COMPILE_OPTIONS of SofaHelper.
        #
        # Exceptionnally, you can prevent a target from being compiled with this
        # option with the SOFA_NO_CXX11 property.  E.g.
        #   set_target_properties(Foo PROPERTIES SOFA_NO_CXX11 1)

        target_compile_options(SofaHelper PUBLIC "$<$<AND:$<NOT:$<BOOL:$<TARGET_PROPERTY:SOFA_NO_CXX11>>>,$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,CXX>>:${CXX11_FLAG}>")
        if ("${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}" MATCHES ".*clang") # Wno-deprecated-register does not exist in GCC
            target_compile_options(SofaHelper PUBLIC "$<$<AND:$<NOT:$<BOOL:$<TARGET_PROPERTY:SOFA_NO_CXX11>>>,$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,CXX>>:-Wno-deprecated-register>")
        endif()
    endif()
endif()

# Eigen
if(MSVC)
    target_compile_definitions(SofaHelper PUBLIC "-DEIGEN_DONT_ALIGN")
endif()

set(SOFAFRAMEWORK_TARGETS SofaCore SofaDefaultType SofaHelper)

## CMake package configurations files
sofa_write_package_config_files(SofaFramework ${SOFAFRAMEWORK_VERSION})

## CMakeParseLibraryList.cmake
# Build tree
configure_file(CMakeParseLibraryList.cmake ${CMAKE_BINARY_DIR}/cmake/CMakeParseLibraryList.cmake COPYONLY)
# Install tree
install(FILES CMakeParseLibraryList.cmake DESTINATION lib/cmake/SofaFramework)

## SofaMacros.cmake
# Build tree
configure_file(SofaMacros.cmake ${CMAKE_BINARY_DIR}/cmake/SofaMacros.cmake COPYONLY)
# Install tree
install(FILES SofaMacros.cmake DESTINATION lib/cmake/SofaFramework)
