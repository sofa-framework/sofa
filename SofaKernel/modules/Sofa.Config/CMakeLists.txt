cmake_minimum_required(VERSION 3.12)
project(Sofa.Config)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

include(SofaMacros)
include(CMakeDependentOption)

message("Adding Sofa.Config")

# Manually define VERSION
set(Sofa_VERSION_MAJOR 20)
set(Sofa_VERSION_MINOR 06)
set(Sofa_VERSION_PATCH 99)
set(Sofa_VERSION ${Sofa_VERSION_MAJOR}.${Sofa_VERSION_MINOR}.${Sofa_VERSION_PATCH})

set(SOFA_VERSION_STR "\"${Sofa_VERSION}\"")
set(SOFA_VERSION "${Sofa_VERSION_MAJOR}${Sofa_VERSION_MINOR}${Sofa_VERSION_PATCH}")


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

# Output Directories definition
set(ARCHIVE_OUTPUT_DIRECTORY lib)
set(RUNTIME_OUTPUT_DIRECTORY bin)
if(WIN32)
    set(LIBRARY_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})
else()
    set(LIBRARY_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY})
endif()
## Set the output directories globally
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIBRARY_OUTPUT_DIRECTORY})

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

include(CompilerOptions)

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
    ${SOFACONFIGSRC_ROOT}/version.h.in
)
set(SOFACONFIG_BUILD_OPTIONS_SRC
    ${SOFACONFIGSRC_ROOT}/config/sharedlibrary_defines.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_deprecated_components.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_dump_visitor.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_opengl.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_threading.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_bbox.h.in
)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES} ${SOFA_BUILD_OPTIONS_SRC})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<INSTALL_INTERFACE:include/extlibs/WinDepPack>")
endif()

# Compile options
if(CMAKE_SYSTEM_NAME STREQUAL Linux AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "NDEBUG")
    # On Windows and MacOS, NDEBUG and _DEBUG are automatically
    # set in the default c/cxx flags of the right configurations
endif()
if(MSVC)
    # Increase Number of Sections in .Obj file
    target_compile_options(${PROJECT_NAME} PUBLIC "/bigobj")
    if(SOFA_VECTORIZE)
        # SSE2 flags
        target_compile_options(${PROJECT_NAME} PUBLIC "/arch:SSE2 /fp:fast")
    endif()
endif()

# CMakeParseLibraryList.cmake
configure_file(cmake/CMakeParseLibraryList.cmake ${CMAKE_BINARY_DIR}/cmake/CMakeParseLibraryList.cmake COPYONLY)
install(FILES cmake/CMakeParseLibraryList.cmake DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)

# SofaMacros*.cmake
file(GLOB macro_files RELATIVE ${CMAKE_CURRENT_LIST_DIR}/cmake LIST_DIRECTORIES FALSE "SofaMacros*.cmake")
foreach(macro_file ${macro_files})
    configure_file(${macro_file} ${CMAKE_BINARY_DIR}/cmake/${macro_file} COPYONLY)
    install(FILES ${macro_file} DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)
endforeach()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
