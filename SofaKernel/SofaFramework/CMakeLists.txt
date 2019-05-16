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

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    set(SOFA_WITH_DEPRECATED_COMPONENTS_ "1")
else()
    set(SOFA_WITH_DEPRECATED_COMPONENTS_ "0")
endif()

if(SOFA_NO_UPDATE_BBOX)
    set(SOFA_NO_UPDATE_BBOX_ "1")
else()
    set(SOFA_NO_UPDATE_BBOX_ "0")
endif()

### Dependencies

set(SOFAFRAMEWORK_DEPENDENCY_LINK) # will be added to SofaHelper link libraries
set(SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES) # will be added to SofaHelper include dirs

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

    if(OPENGL_GLU_FOUND AND UNIX AND NOT APPLE AND SOFA_BUILD_RELEASE_PACKAGE)
        # Add GLU to Linux binaries
        sofa_install_libraries(TARGETS OpenGL::GLU)
    endif()

    ## GLEW
    find_package(GLEW)
    if(GLEW_FOUND)
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK GLEW::GLEW)
        list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})

        sofa_install_libraries(TARGETS GLEW::GLEW)
    else()
        message("OpenGL advanced functions (e.g shaders, FBO) are disabled.")
    endif()
    set(SOFA_HAVE_GLEW ${GLEW_FOUND})
endif()

## tinyxml
find_package(TinyXML REQUIRED)
list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK tinyxml)
set(SOFA_HAVE_TINYXML 1)

## zlib
find_package(ZLIB)
if(ZLIB_FOUND)
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ZLIB::ZLIB)

    if(WIN32)
        sofa_install_libraries(TARGETS ZLIB::ZLIB)
    endif()
endif()
set(SOFA_HAVE_ZLIB ${ZLIB_FOUND})

## Boost
# required boost libraries
find_package(Boost
    REQUIRED system filesystem locale program_options chrono atomic
    OPTIONAL_COMPONENTS thread date_time
    )

set(SOFA_HAVE_BOOST 1)
set(SOFA_HAVE_BOOST_SYSTEM ${Boost_SYSTEM_FOUND})
set(SOFA_HAVE_BOOST_FILESYSTEM ${Boost_FILESYSTEM_FOUND})
set(SOFA_HAVE_BOOST_LOCALE ${Boost_LOCALE_FOUND})
set(SOFA_HAVE_BOOST_PROGRAM_OPTIONS ${Boost_PROGRAM_OPTIONS_FOUND})
set(SOFA_HAVE_BOOST_THREAD ${Boost_THREAD_FOUND})
set(SOFA_HAVE_BOOST_DATE_TIME ${Boost_DATE_TIME_FOUND})

set(BOOST_TARGETS
    Boost::system
    Boost::filesystem
    Boost::locale
    Boost::program_options
    )
if(SOFA_HAVE_BOOST_DATE_TIME)
    list(APPEND BOOST_TARGETS Boost::date_time)
endif()
if(SOFA_HAVE_BOOST_THREAD)
    list(APPEND BOOST_TARGETS Boost::thread)
endif()

list(APPEND Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK ${BOOST_TARGETS})
sofa_install_libraries(
    TARGETS ${BOOST_TARGETS}
    LIBRARIES ${Boost_CHRONO_LIBRARY} ${Boost_ATOMIC_LIBRARY}
    )

## Eigen
find_package(Eigen3 REQUIRED)
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
install(DIRECTORY ${EIGEN3_INCLUDE_DIR}/Eigen
    COMPONENT headers
    DESTINATION "include/extlibs/Eigen"
    PATTERN "*.in" EXCLUDE
    PATTERN "*.txt" EXCLUDE
    PATTERN "*.cpp" EXCLUDE)
install(DIRECTORY ${EIGEN3_INCLUDE_DIR}/unsupported/Eigen
    COMPONENT headers
    DESTINATION "include/extlibs/Eigen"
    PATTERN "*.in" EXCLUDE
    PATTERN "*.txt" EXCLUDE
    PATTERN "*.cpp" EXCLUDE)

## Generate & install the different files that contains the defines associated with
## the state of the build option.
set(SOFA_BUILD_OPTIONS_SRC
        sharedlibrary_defines.h
        build_option_deprecated_components.h
        build_option_dump_visitor.h
        build_option_opengl.h
        build_option_threading.h
        build_option_bbox.h
    )
foreach(NAME ${SOFA_BUILD_OPTIONS_SRC})
    configure_file("${NAME}.in" "${CMAKE_BINARY_DIR}/include/sofa/config/${NAME}")
    install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config/${NAME}" DESTINATION "include/${PROJECT_NAME}/sofa/config" COMPONENT headers)
endforeach()

configure_file("config.h.in" "${CMAKE_BINARY_DIR}/include/sofa/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config.h" DESTINATION "include/${PROJECT_NAME}/sofa" COMPONENT headers)

## Version
set(SOFAFRAMEWORK_VERSION "${Sofa_VERSION}")
set(SOFA_VERSION_STR "\"${Sofa_VERSION}\"")
set(SOFA_VERSION "${Sofa_VERSION_MAJOR}${Sofa_VERSION_MINOR}${Sofa_VERSION_PATCH}")
configure_file(../framework/sofa/version.h.in "${CMAKE_BINARY_DIR}/include/sofa/version.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/version.h" DESTINATION include/${PROJECT_NAME}/sofa COMPONENT headers)


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

if(WIN32)
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
foreach(TARGET ${SOFAFRAMEWORK_TARGETS})
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
endforeach()

## CMake package configurations files
sofa_write_package_config_files(SofaFramework ${SOFAFRAMEWORK_VERSION})

## CMakeParseLibraryList.cmake
# Build tree
configure_file(CMakeParseLibraryList.cmake ${CMAKE_BINARY_DIR}/cmake/CMakeParseLibraryList.cmake COPYONLY)
# Install tree
install(FILES CMakeParseLibraryList.cmake DESTINATION lib/cmake/SofaFramework COMPONENT headers)

## SofaMacros.cmake
# Build tree
configure_file(SofaMacros.cmake ${CMAKE_BINARY_DIR}/cmake/SofaMacros.cmake COPYONLY)
# Install tree
install(FILES SofaMacros.cmake DESTINATION lib/cmake/SofaFramework COMPONENT headers)
