cmake_minimum_required(VERSION 3.12)
project(Sofa.Config LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/lib/cmake")

include(SofaMacros)

# Clear internal target list (set by the macro sofa_add_generic() )
set_property(GLOBAL PROPERTY __GlobalTargetList__ "")
set_property(GLOBAL PROPERTY __GlobalTargetNameList__ "")

# Help RELOCATABLE plugins to resolve their dependencies.
# See SofaMacrosInstall.cmake for usage of this property.
define_property(TARGET
    PROPERTY "RELOCATABLE_INSTALL_DIR"
    BRIEF_DOCS "Install directory of RELOCATABLE target"
    FULL_DOCS "Install directory of RELOCATABLE target"
    )

# Enable the organisation in folders for generators that support
# it. (E.g. some versions of Visual Studio have 'solution folders')
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

## RPATH
if(UNIX)
    # RPATH is a field in ELF binaries that is used as a hint by the system
    # loader to find needed shared libraries.
    #
    # In the build directory, cmake creates binaries with absolute paths in
    # RPATH.  And by default, it strips RPATH from installed binaries.  Here we
    # use CMAKE_INSTALL_RPATH to set a relative RPATH.  By doing so, we avoid
    # the need to play with LD_LIBRARY_PATH to get applications to run.

    # see https://cmake.org/Wiki/CMake_RPATH_handling for $ORIGIN doc
    set(CMAKE_INSTALL_RPATH
        "$ORIGIN/../lib"
        "$$ORIGIN/../lib"
        )

    if(APPLE)
        set(CMAKE_MACOSX_RPATH ON)
        list(APPEND CMAKE_INSTALL_RPATH
            "@loader_path/../lib"
            "@executable_path/../lib"
            "@loader_path/../../Frameworks"
            "@executable_path/../../Frameworks"
            )
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    endif()
endif(UNIX)

### Sofa using type double or float or both
set(SOFA_FLOATING_POINT_TYPE both CACHE STRING
    "Type used for floating point values in SOFA. It actually determines:
    - what template instanciations will be compiled (via the definition of the
    SOFA_FLOAT and SOFA_DOUBLE macros)
    - what is the type behind the 'SReal' typedef used throughout SOFA. (If 'both'
    is selected, SReal defaults to double.)")
set_property(CACHE SOFA_FLOATING_POINT_TYPE PROPERTY STRINGS float double both)

if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_DOUBLE 1)          # Used in sofa/config.h.in
elseif(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_FLOAT 1)           # Used in sofa/config.h.in
endif()

# If you really don't understand the negated logics of SOFA_DOUBLE and
# SOFA_FLOAT please consider using SOFA_WITH_FLOAT and SOFA_WITH_DOUBLE.
# Eg: SOFA_WITH_FLOAT indicate that you need to generate the
# float code and SOFA_WITH_DOUBLE indicates that you
# nedd to generate the double related code.
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_WITH_FLOAT 1)
endif()
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_WITH_DOUBLE 1)
endif()
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL both)
    set(SOFA_WITH_FLOAT 1)
    set(SOFA_WITH_DOUBLE 1)
endif()

# Options
option(SOFA_DETECTIONOUTPUT_FREEMOTION "Compile Sofa with the DETECTIONOUTPUT_FREEMOTION macro defined." OFF)
option(SOFA_NO_OPENGL "Compile Sofa with no OpenGL support. (This will define the SOFA_NO_OPENGL macro.)" OFF)
option(SOFA_NO_UPDATE_BBOX "Compile Sofa with the SOFA_NO_UPDATE_BBOX macro defined." OFF)
option(SOFA_DUMP_VISITOR_INFO "Compile Sofa with the SOFA_DUMP_VISITOR_INFO macro defined." OFF)
if(MSVC)
    option(SOFA_VECTORIZE "Enable the use of SSE2 instructions by the compiler (Only available for MSVC)." OFF)
endif()
### Mask
option(SOFA_USE_MASK "Use mask optimization" OFF)
### SOFA_DEV_TOOL
option(SOFA_WITH_DEVTOOLS "Compile with developement extra features." ON)
### Threading
option(SOFA_WITH_THREADING "Compile sofa with thread-safetiness support (PARTIAL/EXPERIMENTAL)" ON)
### Deprecated components
option(SOFA_WITH_DEPRECATED_COMPONENTS "Compile SOFA with all deprecated components" ON)
if(SOFA_WITH_DEPRECATED_COMPONENTS)
    message(DEPRECATION "Deprecated components are activated (SOFA_WITH_DEPRECATED_COMPONENTS variable is ON)")
endif()

# Variables to expose in configured files
sofa_set_01(SOFA_WITH_THREADING_ VALUE ${SOFA_WITH_THREADING}) # build_option_threading.h.in
sofa_set_01(SOFA_WITH_DEPRECATED_COMPONENTS_ VALUE ${SOFA_WITH_DEPRECATED_COMPONENTS}) # build_option_deprecated_components.h.in
sofa_set_01(SOFA_NO_UPDATE_BBOX_ VALUE ${SOFA_NO_UPDATE_BBOX}) # build_option_bbox.h.in

# Create build and install versions of etc/sofa.ini:
#   - In build dir, sofa.ini contains absolute paths to distant (in source) share/ and examples/ dirs
#   - In install dir, sofa.ini (generated later via installedSofa.ini) contains relative paths to local share/ and examples/ dirs
set(SHARE_DIR "${CMAKE_SOURCE_DIR}/share")
set(EXAMPLES_DIR "${CMAKE_SOURCE_DIR}/examples")
configure_file("etc/sofa.ini.in" "${CMAKE_BINARY_DIR}/etc/sofa.ini")
set(SHARE_DIR "../share/sofa")
set(EXAMPLES_DIR "../share/sofa/examples")
configure_file("etc/sofa.ini.in" "${CMAKE_BINARY_DIR}/etc/installedSofa.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini" DESTINATION etc RENAME sofa.ini COMPONENT applications)

set(SOFACONFIGSRC_ROOT "src/sofa")
set(HEADER_FILES
    ${SOFACONFIGSRC_ROOT}/config.h.in
    ${SOFACONFIGSRC_ROOT}/config/qt.conf.h
    ${SOFACONFIGSRC_ROOT}/version.h.in
)
set(SOURCE_FILES
    ${SOFACONFIGSRC_ROOT}/initSofaConfig.cpp # necessary to build a library
    )
set(SOFACONFIG_BUILD_OPTIONS_SRC
    ${SOFACONFIGSRC_ROOT}/config/sharedlibrary_defines.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_deprecated_components.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_dump_visitor.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_opengl.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_threading.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_bbox.h.in
)

# Header-only library (not handled well by macros & stuff so commented)
# add_library(${PROJECT_NAME} INTERFACE)
# target_sources(${PROJECT_NAME} INTERFACE "$<BUILD_INTERFACE:${HEADER_FILES};${SOFACONFIG_BUILD_OPTIONS_SRC}>")

# Library
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOFACONFIG_BUILD_OPTIONS_SRC} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<INSTALL_INTERFACE:include/extlibs/WinDepPack>")
endif()

# Compile options
# Set C++17
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set(SOFACONFIG_COMPILE_OPTIONS_DEBUG "")
set(SOFACONFIG_COMPILE_OPTIONS_RELEASE "")
set(SOFACONFIG_COMPILE_OPTIONS "")
set(SOFACONFIG_LINK_OPTIONS "")
set(SOFACONFIG_LINK_OPTIONS_DEBUG "")
set(SOFACONFIG_LINK_OPTIONS_RELEASE "")

## GCC-specific
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    ## Find out the version of g++ (and save it in GCXX_VERSION)
    if(CMAKE_CXX_COMPILER_ARG1) # CXX="ccache g++"
        string(STRIP ${CMAKE_CXX_COMPILER_ARG1} CMAKE_CXX_COMPILER_ARG1_stripped)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER_ARG1_stripped} -dumpversion OUTPUT_VARIABLE GCXX_VERSION)
    else()
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCXX_VERSION)
    endif()

    ## Disable partial inlining under gcc 4.6 (Why? -> the memory was exploding)
    if("${GCXX_VERSION}" VERSION_EQUAL 4.6)
        list(APPEND SOFACONFIG_COMPILE_OPTIONS_RELEASE "-fno-partial-inlining")
    endif()

    # stack-protector
    list(APPEND SOFACONFIG_COMPILE_OPTIONS_RELEASE "-fstack-protector;--param=ssp-buffer-size=4")
    # _FORTIFY_SOURCE
    list(APPEND SOFACONFIG_COMPILE_OPTIONS_RELEASE "-D_FORTIFY_SOURCE=2")

    if(CMAKE_BUILD_TYPE MATCHES "Release")
        # ???
        list(APPEND SOFACONFIG_LINK_OPTIONS "-Wl,--no-undefined;-lc")
    endif()

endif()

## GCC/Clang-specific
if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # Warnings
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "-Wall;-W;-Wno-padded")

    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PUBLIC -Wno-attributes)
endif()

## Windows-specific
if(WIN32)    
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "-D_USE_MATH_DEFINES")
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "/MP;/wd4250;/wd4251;/wd4275;/wd4675;/wd4996;/wd4661")
    # 4661: no suitable definition provided for explicit template instantiation request
    # it happens because we put explicit instantiation in a separate translation unit
    # it is by design, so this warning is irrelevant in our project

    if(MSVC_TOOLSET_VERSION GREATER 140) # > VS 2015
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "/Zc:__cplusplus")
    endif()

    # Focus on max speed in Release Mode with link-time optimization
    option(SOFA_ENABLE_LINK_TIME_OPTIMIZATION "Enable LTCG IN release mode (MSVC only for now) [Warning, use a lot of disk space!]" OFF)

    if(SOFA_ENABLE_LINK_TIME_OPTIMIZATION)
        list(APPEND SOFACONFIG_COMPILE_OPTIONS_RELEASE "/GL")
        list(APPEND SOFACONFIG_LINK_OPTIONS_RELEASE "/LTCG")
    endif()
endif()

# Mac specific
if(APPLE)
    #remove OpenGL deprecation message
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "-DGL_SILENCE_DEPRECATION")
endif()


## OpenMP
option(SOFA_OPENMP "Compile Sofa with OpenMP multithreading." OFF)
if(SOFA_OPENMP)
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}")
        list(APPEND SOFACONFIG_LINK_OPTIONS "${OpenMP_CXX_FLAGS}")
    else()
        message("WARNING: Your compiler does not implement OpenMP.")
    endif()
endif()

# An important C++11 feature may be not enabled due to
# the compiler being built without the --enable-libstdcxx-time option.
if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "-D_GLIBCXX_USE_NANOSLEEP -D_GLIBCXX_USE_SCHED_YIELD")
    endif()
endif()

# hack for clang on old macosx (version < 10.9, such as the dashboard servers)
# that is using, by default at that time, a libstdc++ that did not fully implement c++11
if(APPLE AND ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND CMAKE_SYSTEM_VERSION VERSION_LESS "10.9" AND ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang" )
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "-stdlib=libc++")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Linux AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "NDEBUG")
    # On Windows and MacOS, NDEBUG and _DEBUG are automatically
    # set in the default c/cxx flags of the right configurations
endif()
if(MSVC)
    # Increase Number of Sections in .Obj file
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "/bigobj")
    if(SOFA_VECTORIZE)
        # SSE2 flags
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "/arch:SSE2;/fp:fast")
    endif()
endif()

# Use Release flags for MinSizeRel and RelWithDebInfo build types:
set_target_properties(${PROJECT_NAME} PROPERTIES
  MAP_IMPORTED_CONFIG_MINSIZEREL Release
  MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
  )
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:RELEASE>:${SOFACONFIG_COMPILE_OPTIONS_RELEASE}>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:DEBUG>:${SOFACONFIG_COMPILE_OPTIONS_DEBUG}>")
target_compile_options(${PROJECT_NAME} PUBLIC "${SOFACONFIG_COMPILE_OPTIONS}")

# target_link_option only available with cmake >= 3.13
if(${CMAKE_VERSION} VERSION_LESS "3.13.0")
    if(NOT SOFACONFIG_LINK_OPTIONS_RELEASE STREQUAL "")
        string(REPLACE ";" " " SOFACONFIG_LINK_OPTIONS_RELEASE "${SOFACONFIG_LINK_OPTIONS_RELEASE}")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE "${SOFACONFIG_LINK_OPTIONS_RELEASE}")
    endif()
    if(NOT SOFACONFIG_LINK_OPTIONS_DEBUG STREQUAL "")
        string(REPLACE ";" " " SOFACONFIG_LINK_OPTIONS_DEBUG "${SOFACONFIG_LINK_OPTIONS_DEBUG}")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG "${SOFACONFIG_LINK_OPTIONS_DEBUG}")
    endif()
    if(NOT SOFACONFIG_LINK_OPTIONS STREQUAL "")
        string(REPLACE ";" " " SOFACONFIG_LINK_OPTIONS "${SOFACONFIG_LINK_OPTIONS}")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${SOFACONFIG_LINK_OPTIONS}")
    endif()
else()
    target_link_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:RELEASE>:${SOFACONFIG_LINK_OPTIONS_RELEASE}>")
    target_link_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:DEBUG>:${SOFACONFIG_LINK_OPTIONS_DEBUG}>")
    target_link_options(${PROJECT_NAME} PUBLIC "${SOFACONFIG_LINK_OPTIONS}")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER SofaFramework) # IDE folder

# Attach Sofa Version into properties
set_target_properties(${PROJECT_NAME} PROPERTIES Sofa_VERSION "${Sofa_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES SOFA_VERSION_STR "${SOFA_VERSION_STR}")
set_target_properties(${PROJECT_NAME} PROPERTIES SOFA_VERSION "${SOFA_VERSION}")

# CMakeParseLibraryList.cmake
configure_file(cmake/CMakeParseLibraryList.cmake ${CMAKE_BINARY_DIR}/cmake/CMakeParseLibraryList.cmake COPYONLY)
install(FILES cmake/CMakeParseLibraryList.cmake DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)

# SofaMacros*.cmake
file(GLOB macro_files RELATIVE ${CMAKE_CURRENT_LIST_DIR}/cmake LIST_DIRECTORIES FALSE "SofaMacros*.cmake")
foreach(macro_file ${macro_files})
    configure_file(${macro_file} ${CMAKE_BINARY_DIR}/cmake/${macro_file} COPYONLY)
    install(FILES ${macro_file} DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)
endforeach()

# own include dir, the macro does not handle "independent" package for SofaFramework-style include 
# i.e NOT the name of package first (sofa/ is wanted, instead of Sofa.Config/sofa)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/${PROJECT_NAME}>")

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )

# Propagate variables (which were in Sofa's global cmakelist)
# TODO: Put instead into a target_property ?
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
set(SOFA_WITH_FLOAT "${SOFA_WITH_FLOAT}" PARENT_SCOPE)
set(SOFA_WITH_DOUBLE "${SOFA_WITH_DOUBLE}" PARENT_SCOPE)
