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

### Sofa using type double or float
set(SOFA_FLOATING_POINT_TYPE double CACHE STRING
    "Type used for floating point values in SOFA. It actually determines:
    - what template instanciations will be compiled (via the definition of the
    SOFA_FLOAT and SOFA_DOUBLE macros)
    - what is the type behind the 'SReal' typedef used throughout SOFA.")
set_property(CACHE SOFA_FLOATING_POINT_TYPE PROPERTY STRINGS float double)

if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_DOUBLE 1)
    set(SOFA_FLOAT 0)
elseif(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_DOUBLE 0)
    set(SOFA_FLOAT 1)
endif()

# If you really don't understand the negated logics of SOFA_DOUBLE and
# SOFA_FLOAT please consider using SOFA_WITH_FLOAT and SOFA_WITH_DOUBLE.
# Eg: SOFA_WITH_FLOAT indicate that you need to generate the
# float code and SOFA_WITH_DOUBLE indicates that you
# need to generate the double related code.
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL float)
    set(SOFA_WITH_FLOAT 1)
    set(SOFA_WITH_DOUBLE 0)
endif()
if(${SOFA_FLOATING_POINT_TYPE} STREQUAL double)
    set(SOFA_WITH_DOUBLE 1)
    set(SOFA_WITH_FLOAT 0)
endif()

# Options
option(SOFA_DETECTIONOUTPUT_FREEMOTION "Compile Sofa with the DETECTIONOUTPUT_FREEMOTION macro defined." OFF)
option(SOFA_NO_UPDATE_BBOX "Compile Sofa with the SOFA_NO_UPDATE_BBOX macro defined." OFF)
option(SOFA_DUMP_VISITOR_INFO "Compile Sofa with the SOFA_DUMP_VISITOR_INFO macro defined." OFF)
if(MSVC)
    option(SOFA_VECTORIZE "(deprecated) Enable the use of SSE2 instructions by the compiler (Only available for MSVC)." OFF)
    option(SOFA_ENABLE_SIMD "Enable the use of SIMD instructions by the compiler (AVX/AVX2 for msvc, march=native for gcc/clang)." OFF)
endif()
option(SOFA_ENABLE_FAST_MATH "Enable floating-point model to fast (theoretically faster but can bring unexpected results/bugs)." OFF)


### SOFA_DEV_TOOL
option(SOFA_WITH_DEVTOOLS "Compile with development extra features." ON)

# Variables to expose in configured files
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
set(SOURCE_FILES
    ${SOFACONFIGSRC_ROOT}/initSofaConfig.cpp # necessary to build a library
    )
set(SOFACONFIG_BUILD_OPTIONS_SRC
    ${SOFACONFIGSRC_ROOT}/config/sharedlibrary_defines.h.in
    ${SOFACONFIGSRC_ROOT}/config/build_option_dump_visitor.h.in
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

## Link-time optimization
if(NOT IPO_CHECK_DONE)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IS_IPO_SUPPORTED OUTPUT IPO_ERROR)
    set(IPO_CHECK_DONE ON CACHE INTERNAL "IPO check has been done" FORCE)
    if (IS_IPO_SUPPORTED)
        set(IPO_SUPPORTED ON CACHE INTERNAL "IPO can be activated" FORCE)
    else()
        message(STATUS "IPO / LTO not supported: <${IPO_ERROR}>")
    endif()
endif()

if (IPO_SUPPORTED)
    # Focus on max speed with link-time optimization
    option(SOFA_ENABLE_LINK_TIME_OPTIMIZATION "Enable interprocedural optimization" OFF)
    if (SOFA_ENABLE_LINK_TIME_OPTIMIZATION)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE PARENT_SCOPE)
    endif()
else()
    message(STATUS "IPO / LTO not supported: <${IPO_ERROR}>")
endif()

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

    if(SOFA_ENABLE_FAST_MATH)
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "-ffast-math")
    endif()
endif()

## Windows-specific
if(WIN32)
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "-D_USE_MATH_DEFINES")
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "-D_CRT_SECURE_NO_WARNINGS")
    list(APPEND SOFACONFIG_COMPILE_OPTIONS "-D_CRT_NONSTDC_NO_DEPRECATE")

    set(SOFA_MSVC_NB_COMPILER_PROCESSES "0" CACHE STRING "The maximum number of processes that the compiler can create. 0 (default) for the number of effective processors on the computer")
    if (SOFA_MSVC_NB_COMPILER_PROCESSES EQUAL "0")
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "/MP")
    else()
        if (SOFA_MSVC_NB_COMPILER_PROCESSES MATCHES "^[0-9]+$")
            list(APPEND SOFACONFIG_COMPILE_OPTIONS "/MP${SOFA_MSVC_NB_COMPILER_PROCESSES}")
        else()
            message(FATAL_ERROR "A number is expected for SOFA_MSVC_NB_COMPILER_PROCESSES")
        endif()
    endif()

    list(APPEND SOFACONFIG_COMPILE_OPTIONS "/wd4250;/wd4251;/wd4275;/wd4675;/wd4661")
    # 4661: no suitable definition provided for explicit template instantiation request
    # it happens because we put explicit instantiation in a separate translation unit
    # it is by design, so this warning is irrelevant in our project

    if(MSVC_TOOLSET_VERSION GREATER 140) # > VS 2015
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "/Zc:__cplusplus")
    endif()

    # Experimental: compilation with MSVC/Clang-cl
    if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        if( ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT} STREQUAL MSVC) # something weird happens if we put the two test in one if ??
            message(WARNING "Experimental: you are trying to compile with MSVC and the clang-cl toolchain; this is not officially supported.")
            # remove lots of warnings (the Wall of "normal" clang seems different of Wall of clang-cl)
            list(APPEND SOFACONFIG_COMPILE_OPTIONS -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-double-promotion -Wno-old-style-cast -Wno-reserved-id-macro -Wno-language-extension-token -Wno-dllexport-explicit-instantiation-decl -Wno-nonportable-system-include-path -Wno-zero-as-null-pointer-constant -Wno-documentation)
            # optimization flags (not sure if necessary..)
            list(APPEND SOFACONFIG_COMPILE_OPTIONS_RELEASE -march=native)
            if(SOFA_ENABLE_FAST_MATH)
                list(APPEND SOFACONFIG_COMPILE_OPTIONS "/fp:fast")
            endif()
        endif()
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
    sofa_find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}")
        list(APPEND SOFACONFIG_LINK_OPTIONS "${OpenMP_CXX_FLAGS}")
    else()
        message("WARNING: Your compiler does not implement OpenMP.")
    endif()
endif()

## Tracy
option(SOFA_TRACY "Compile SOFA with the Tracy profiler client" OFF)
if (SOFA_TRACY)
    set(SOFA_TRACY_VERSION v0.9.1)
    include(FetchContent)
    option(TRACY_STATIC "Whether to build Tracy as a static library" OFF)
    FetchContent_Declare (
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG ${SOFA_TRACY_VERSION}
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable (tracy)
    target_link_libraries(${PROJECT_NAME} PUBLIC TracyClient )
    message(STATUS "SOFA is compiled with the Tracy profiler client. Use the Tracy server ${SOFA_TRACY_VERSION}.")
endif()


option(SOFA_ENABLE_BUILTIN_TIMER "Enable the builtin timers" ON)
if(SOFA_ENABLE_BUILTIN_TIMER)
    set(SOFA_ENABLE_SCOPED_ADVANCED_TIMER 1)
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
        message("SOFA_VECTORIZE option is deprecated and has been split between SOFA_ENABLE_SIMD and SOFA_ENABLE_FAST_MATH. These 2 options will be enabled.")
        set(SOFA_ENABLE_SIMD ON)
        set(SOFA_ENABLE_FAST_MATH ON)
    endif()
    # SIMD flags
    if(SOFA_ENABLE_SIMD)
        include(find_avx)
        MSVC_CHECK_FOR_AVX() # search for availability of AVX2 (otherwise set to AVX)
        if (HAVE_AVX2_EXTENSIONS)
            message("Your CPU supports AVX2 SIMD. Using flags: ${AVX_FLAGS}")
        elseif(HAVE_AVX_EXTENSIONS)
            message("Your CPU supports only AVX SIMD. Using flags: ${AVX_FLAGS}")
        else()
            message("Your CPU does not support AVX SIMD.") # SSE2 is the default when x64 is set
        endif()
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "${AVX_FLAGS}")
    endif()
    if(SOFA_ENABLE_FAST_MATH)
        list(APPEND SOFACONFIG_COMPILE_OPTIONS "/fp:fast")
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

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

# Attach Sofa Version into properties
set_target_properties(${PROJECT_NAME} PROPERTIES Sofa_VERSION "${Sofa_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES SOFA_VERSION_STR "${SOFA_VERSION_STR}")
set_target_properties(${PROJECT_NAME} PROPERTIES SOFA_VERSION "${SOFA_VERSION}")

# CMakeParseLibraryList.cmake
configure_file(cmake/CMakeParseLibraryList.cmake ${CMAKE_BINARY_DIR}/lib/cmake/CMakeParseLibraryList.cmake COPYONLY)
install(FILES cmake/CMakeParseLibraryList.cmake DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)

# SofaMacros*.cmake
set(macro_files SofaMacros.cmake SofaMacrosConfigure.cmake SofaMacrosInstall.cmake SofaMacrosPython.cmake SofaMacrosUtils.cmake)
foreach(macro_file ${macro_files})
    configure_file(cmake/${macro_file} ${CMAKE_BINARY_DIR}/lib/cmake/${macro_file} COPYONLY)
    install(FILES cmake/${macro_file} DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)
endforeach()

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
