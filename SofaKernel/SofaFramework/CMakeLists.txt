cmake_minimum_required(VERSION 3.1)
project(SofaFramework)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include(SofaMacros)
# Clear internal target list (set by the macro sofa_add_generic() )
set_property(GLOBAL PROPERTY __GlobalTargetList__ "")
set_property(GLOBAL PROPERTY __GlobalTargetNameList__ "")

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

if(MSVC)
    option(SOFA_VECTORIZE "Enable the use of SSE2 instructions by the compiler (Only available for MSVC)." OFF)
endif()

if(SOFA_WITH_THREADING)
    set(SOFA_WITH_THREADING_ "1")
else()
    set(SOFA_WITH_THREADING_ "0")
endif()

if(SOFA_WITH_EXPERIMENTAL_FEATURES)
    set(SOFA_WITH_EXPERIMENTAL_FEATURES_ "1")
else()
    set(SOFA_WITH_EXPERIMENTAL_FEATURES_ "0")
endif()

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    set(SOFA_WITH_DEPRECATED_COMPONENTS_ "1")
else()
    set(SOFA_WITH_DEPRECATED_COMPONENTS_ "0")
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

## Boost
# required boost libraries
find_package(Boost
    REQUIRED system filesystem locale program_options
    OPTIONAL_COMPONENTS thread date_time
    )

set(SOFA_HAVE_BOOST_SYSTEM ${Boost_SYSTEM_FOUND})
set(SOFA_HAVE_BOOST_FILESYSTEM ${Boost_FILESYSTEM_FOUND})
set(SOFA_HAVE_BOOST_LOCALE ${Boost_LOCALE_FOUND})
set(SOFA_HAVE_BOOST_PROGRAM_OPTIONS ${Boost_PROGRAM_OPTIONS})
set(SOFA_HAVE_BOOST_THREAD ${Boost_THREAD_FOUND})
set(SOFA_HAVE_BOOST_DATE_TIME ${Boost_DATE_TIME_FOUND})

list(APPEND Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})

if(SOFA_HAVE_BOOST_SYSTEM
   AND SOFA_HAVE_BOOST_FILESYSTEM
   AND SOFA_HAVE_BOOST_LOCALE)

    set(SOFA_HAVE_BOOST 1)

    sofa_create_target(BoostSystem SofaFramework "${Boost_SYSTEM_LIBRARY}" "${Boost_INCLUDE_DIRS}")
    sofa_create_target(BoostFileSystem SofaFramework "${Boost_FILESYSTEM_LIBRARY}" "${Boost_INCLUDE_DIRS}")
    sofa_create_target(BoostLocale SofaFramework "${Boost_LOCALE_LIBRARY}" "${Boost_INCLUDE_DIRS}")
    sofa_create_target(BoostProgramOptions SofaFramework "${Boost_PROGRAM_OPTIONS_LIBRARY}" "${Boost_INCLUDE_DIRS}")

    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostSystem_Target})
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostFileSystem_Target})
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostLocale_Target})
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostProgramOptions_Target})

    sofa_install_get_libraries(${Boost_SYSTEM_LIBRARY})
    sofa_install_get_libraries(${Boost_LOCALE_LIBRARY})
    sofa_install_get_libraries(${Boost_FILESYSTEM_LIBRARY})
    sofa_install_get_libraries(${Boost_PROGRAM_OPTIONS_LIBRARY})

    if(SOFA_HAVE_BOOST_DATE_TIME)
        sofa_create_target(BoostDateTime SofaFramework "${Boost_DATE_TIME_LIBRARY}" "${Boost_INCLUDE_DIRS}")
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostDateTime_Target})
        sofa_install_get_libraries(${Boost_DATE_TIME_LIBRARY})
    endif()

    if(SOFA_HAVE_BOOST_THREAD)
        sofa_create_target(BoostThread SofaFramework "${Boost_THREAD_LIBRARY}" "${Boost_INCLUDE_DIRS}")
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BoostThread_Target})
        sofa_install_get_libraries(${Boost_THREAD_LIBRARY})
    endif()

else()
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif()

## Eigen
find_package(Eigen3 REQUIRED)
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})

## JSON
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${JSON_INCLUDE_DIR})

## Generate & install the different files that contains the defines associated with
## the state of the build option.
set(SOFA_BUILD_OPTIONS_SRC
        sharedlibrary_defines.h
        build_option_deprecated_components.h
        build_option_dump_visitor.h
        build_option_experimental_features.h
        build_option_opengl.h
        build_option_threading.h
    )
foreach(NAME ${SOFA_BUILD_OPTIONS_SRC})
    configure_file("${NAME}.in" "${CMAKE_BINARY_DIR}/include/sofa/config/${NAME}")
    install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config/${NAME}" DESTINATION "include/sofa/config")
endforeach()

configure_file("config.h.in" "${CMAKE_BINARY_DIR}/include/sofa/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config.h" DESTINATION "include/sofa/")

# make sure everyone in the build tree can see <sofa/config.h>
#list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR})
list(REMOVE_DUPLICATES SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES)

## Version
set(SOFAFRAMEWORK_VERSION "18.06.dev")

## sofa/version.h
set(SOFA_VERSION "180699")
set(SOFA_VERSION_STR "\"18.06.dev\"")
configure_file(../framework/sofa/version.h.in "${CMAKE_BINARY_DIR}/include/sofa/version.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/version.h" DESTINATION include/sofa)


## Subdirectories
add_subdirectory(../framework/sofa/helper ${CMAKE_CURRENT_BINARY_DIR}/SofaHelper)
add_subdirectory(../framework/sofa/defaulttype ${CMAKE_CURRENT_BINARY_DIR}/SofaDefaultType)
add_subdirectory(../framework/sofa/core ${CMAKE_CURRENT_BINARY_DIR}/SofaCore)
add_subdirectory(../framework/sofa/simulation ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationCore)
if(SOFA_BUILD_TESTS)
    add_subdirectory(../framework/framework_test ${CMAKE_CURRENT_BINARY_DIR}/framework_test)
endif()

# IDE folder
set_target_properties(SofaHelper PROPERTIES FOLDER "SofaFramework")
set_target_properties(SofaDefaultType PROPERTIES FOLDER "SofaFramework")
set_target_properties(SofaCore PROPERTIES FOLDER "SofaFramework")
set_target_properties(SofaSimulationCore PROPERTIES FOLDER "SofaFramework")

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
