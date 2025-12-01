cmake_minimum_required(VERSION 3.22)
project(Sofa
    HOMEPAGE_URL https://www.sofa-framework.org/
) # Cannot use VERSION with patch like "00"

include(CMakeDependentOption)


if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0") 
    set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)
    # https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_PACKAGE_TARGETS_GLOBAL.html
endif()

# Manually define VERSION
set(Sofa_VERSION_MAJOR 25)
set(Sofa_VERSION_MINOR 12)
set(Sofa_VERSION_PATCH 99)
set(Sofa_VERSION ${Sofa_VERSION_MAJOR}.${Sofa_VERSION_MINOR}.${Sofa_VERSION_PATCH})

set(SOFA_URL "${CMAKE_PROJECT_HOMEPAGE_URL}")

set(SOFA_VERSION_STR "\"${Sofa_VERSION}\"")
set(SOFA_VERSION "${Sofa_VERSION_MAJOR}${Sofa_VERSION_MINOR}${Sofa_VERSION_PATCH}")

## Default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to Release as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

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

set(SOFA_APPLICATIONS_RELOCATABLE_PATH "applications" CACHE STRING "Relocatable path that will be used ot install applications")
set(SOFA_PLUGINS_RELOCATABLE_PATH "plugins" CACHE STRING "Relocatable path that will be used ot install plugins")
set(SOFA_PROJECTS_RELOCATABLE_PATH "projects" CACHE STRING "Relocatable path that will be used ot install projects")



# Option for packaging
option(SOFA_BUILD_RELEASE_PACKAGE "Run package specific configure" OFF)

# Option to allow some dependencies such as cxxopts to be fetched by cmake if
# the package is not found
option(SOFA_ALLOW_FETCH_DEPENDENCIES "Allow compatible dependencies to be fetched if the package is not found by cmake.
  List of dependencies that can be fetched: cxxopts, gtest, metis, CImg, nlohmann json" ON)

# Option to accelerate the compilation
# see https://cmake.org/cmake/help/v3.16/command/target_precompile_headers.html
# and https://cmake.org/cmake/help/v3.16/prop_tgt/DISABLE_PRECOMPILE_HEADERS.html

# https://cmake.org/cmake/help/latest/policy/CMP0127.html
cmake_policy(SET CMP0127 NEW)

option(SOFA_BUILD_WITH_PCH_ENABLED "Compile SOFA using precompiled header (to accelerate the build process)" OFF)
if(SOFA_BUILD_WITH_PCH_ENABLED)
    message("-- Precompiled headers: enabled (SOFA_BUILD_WITH_PCH_ENABLED is ON).")
else()
    message("-- Precompiled headers: disabled (SOFA_BUILD_WITH_PCH_ENABLED is OFF).")
    set(DISABLE_PRECOMPILE_HEADERS ON)
endif()

## Change default install component and prefix
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME applications)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
endif()
if(APPLE AND SOFA_BUILD_APP_BUNDLE)
    set(SOFA_BUILD_RELEASE_PACKAGE ON)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/runSofa.app/Contents/MacOS)
    message("SOFA_BUILD_APP_BUNDLE = ${SOFA_BUILD_APP_BUNDLE}\n"
        "  Forcing SOFA_BUILD_RELEASE_PACKAGE = ${SOFA_BUILD_RELEASE_PACKAGE}\n"
        "  Forcing CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}"
        )
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Remove generated CMake files, this prevents CMake from finding packages that
# were disabled (like, unchecked in cmake-gui) after being built once.
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/cmake)
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/lib/cmake)
# Remove generated SofaPython configuration files, to prevent SofaPython from
# adding paths to disabled plugins.
file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/etc/sofa/python.d)

## Custom Environment
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Sofa/framework/Config/cmake")
include(SofaMacros)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindEigen3.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/macdeployqt.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/windeployqt.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/FindTinyXML2.cmake"
    DESTINATION lib/cmake/Modules
    COMPONENT headers
    )

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


### Windows config
if(MSVC)
    option(SOFA_USE_DEPENDENCY_PACK "Enable the use of the Windows Dependency Pack" ON)
    # WinDepPack
    if(SOFA_USE_DEPENDENCY_PACK)
        set(SOFA_DEPENDENCY_PACK_DIR "" CACHE PATH "Directory containing Windows Dependency Pack")
        if(SOFA_DEPENDENCY_PACK_DIR AND EXISTS ${SOFA_DEPENDENCY_PACK_DIR})
            add_subdirectory(${SOFA_DEPENDENCY_PACK_DIR} ${CMAKE_BINARY_DIR}/extlibs/SofaDependencyPack)
        else ()
            message(INFO "Fetching WinDepPack")
            sofa_fetch_dependency(WinDepPack GIT_REPOSITORY "https://github.com/sofa-framework/WinDepPack.git" GIT_TAG "master" DONT_BUILD)
            add_subdirectory("${CMAKE_BINARY_DIR}/external_directories/fetched/WinDepPack" "${CMAKE_BINARY_DIR}/extlibs/WinDepPack")
        endif()
    endif ()
endif()

### Testing
option(SOFA_BUILD_TESTS "Compile the automatic tests for Sofa, along with the gtest library." OFF)

## Active or not the use of ccache
option(SOFA_USE_CCACHE "Compile using ccache optimization" OFF)
if(SOFA_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        # Support Unix Makefiles and Ninja
        #TODO replace by <LANG>_COMPILER_LAUNCHER when min cmake version > 3.4
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    else()
        message(WARNING "ccache not found, disabling option")
        set(SOFA_USE_CCACHE OFF CACHE bool "Compile using ccache optimization" FORCE)
    endif()
endif()

### Ninja build pools
option(SOFA_NINJA_BUILD_POOLS "Activate the Ninja build pools feature, to limit the cores used by specific targets" OFF)

### Warn the user that the legacy option has been deleted and is useless now
if(DEFINED SOFA_ENABLE_LEGACY_HEADERS)
    message(WARNING "SOFA_ENABLE_LEGACY_HEADERS cmake option has been deleted in v24.12 and is now meaningless.")
endif()
### Warn the user that HeadlessRecorder option has been renamed (due to its move into applications/plguins)
if(DEFINED PLUGIN_SOFA_GUI_HEADLESSRECORDER AND PLUGIN_SOFA_GUI_HEADLESSRECORDER)
    message(WARNING "HeadlessRecorder has been moved to applications/plugins in v24.12, thus its cmake option is named PLUGIN_HEADLESSRECORDER now.")
endif()


# Sofa.Config sets the environment (options, compiler flags, global variables)
sofa_add_subdirectory(library Sofa/framework/Config Sofa.Config ON)

### Extlibs
add_subdirectory(extlibs)

### SOFA (framework and components)
add_subdirectory(Sofa)

# SceneCreator plugin
# Library used by SOFA_BUILD_TESTS
sofa_add_subdirectory(plugin applications/plugins/SceneCreator SceneCreator OFF
    WHEN_TO_SHOW "NOT SOFA_BUILD_SCENECREATOR AND NOT SOFA_BUILD_TESTS AND NOT SOFA_BUILD_RELEASE_PACKAGE"
    VALUE_IF_HIDDEN "ON")

## add SofaPython3, guis, plugins, projects
add_subdirectory(applications)

## SimpleAPI examples
option(SOFA_BUILD_SIMPLEAPI_EXAMPLES "Build SOFA scenes based on SimpleAPI." OFF)
if(SOFA_BUILD_SIMPLEAPI_EXAMPLES)
    add_subdirectory(examples/SimpleAPI)
endif()


## SOFA scenes
option(SOFA_BUILD_ADD_SCENES "Add SOFA scenes as a project of the build." OFF)
if (SOFA_BUILD_ADD_SCENES)
    add_subdirectory(examples)
endif()

## SOFA shaders
option(SOFA_BUILD_ADD_SHADERS "Add SOFA shaders as a project of the build." OFF)
if (SOFA_BUILD_ADD_SHADERS)
    add_subdirectory(share/shaders)
endif()

## Build external projects at the same time
set(SOFA_EXTERNAL_DIRECTORIES "" CACHE STRING "list of paths separated by ';'")
if(NOT "${SOFA_EXTERNAL_DIRECTORIES}" STREQUAL "")
    foreach(dir ${SOFA_EXTERNAL_DIRECTORIES})
        get_filename_component(name ${dir} NAME) # Get the name of the actual directory
        message("Adding external directory: ${name} (${dir})")
        add_subdirectory(${dir} "${CMAKE_CURRENT_BINARY_DIR}/external_directories/${name}")
    endforeach()
endif()


## IDE directories
# Sofa.Component
sofa_get_all_targets(all_targets)
foreach(target ${all_targets})
    if(target MATCHES "Sofa.Component.*" AND NOT target MATCHES ".*_relocatable_install")
        set_target_properties(${target} PROPERTIES FOLDER Sofa.Component) # IDE folder
    endif()
endforeach()
# Testing
if(TARGET Sofa.Testing) # No need to search for test targets if tests are not enabled
    sofa_get_all_targets(all_targets)
    foreach(target ${all_targets})
        if(target MATCHES ".*_test" OR target MATCHES ".*_simutest")
            set_target_properties(${target} PROPERTIES FOLDER Testing) # IDE folder
        endif()
    endforeach()
endif()
##

## Install configuration
#install(FILES "${CMAKE_BINARY_DIR}/CMakeCache.txt" DESTINATION . COMPONENT headers)
install(FILES "${CMAKE_SOURCE_DIR}/README.md" DESTINATION . COMPONENT applications)
install(FILES "${CMAKE_SOURCE_DIR}/CHANGELOG.md" DESTINATION . COMPONENT applications)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE-LGPL.md" DESTINATION . COMPONENT applications)
install(FILES "${CMAKE_SOURCE_DIR}/Authors.txt" DESTINATION . COMPONENT applications)

option(SOFA_INSTALL_RESOURCES_FILES "Copy resources files (share/, examples/) when installing" ON)
## Install resource files
if(SOFA_INSTALL_RESOURCES_FILES)
    install(DIRECTORY share/ DESTINATION share/sofa COMPONENT resources)
    install(DIRECTORY examples/ DESTINATION share/sofa/examples COMPONENT resources)
endif()

file(WRITE "${CMAKE_BINARY_DIR}/plugins/README.txt"
    "This folder will be automatically scanned by the Plugin Manager.\n"
    "For all info about plugins, see https://www.sofa-framework.org/community/doc/using-sofa/lexicography/#plugin")
install(DIRECTORY ${CMAKE_BINARY_DIR}/plugins/ DESTINATION plugins COMPONENT resources)
file(WRITE "${CMAKE_BINARY_DIR}/collections/README.txt"
    "This folder will be automatically scanned by the Plugin Manager.\n"
    "For all info about collections, see https://www.sofa-framework.org/community/doc/using-sofa/lexicography/#collection")
install(DIRECTORY ${CMAKE_BINARY_DIR}/collections/ DESTINATION collections COMPONENT resources)

sofa_install_git_infos(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR})

# Post-install scripts (must be the last add_subdirectory)
add_subdirectory(tools/postinstall-fixup)

if(SOFA_BUILD_RELEASE_PACKAGE)
    #######################
    # CPack configuration #
    #######################

    ################
    # Package config
    include(CPackComponent)
    include(CPackIFW)
    set(CPACK_PACKAGE_VERSION "${Sofa_VERSION}")
    set(CPACK_PACKAGE_NAME "SOFA v${CPACK_PACKAGE_VERSION}")
    set(CPACK_PACKAGE_VENDOR "The SOFA Team")
    set(CPACK_PACKAGE_CONTACT "contact@sofa-framework.org")
    set(CPACK_PACKAGE_DESCRIPTION "Real-time multi-physics simulation with an emphasis on medical simulation.")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Real-time multi-physics simulation with an emphasis on medical simulation.")
    set(CPACK_PACKAGE_EXECUTABLES "runSofa" "runSofa")
    set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
    set(CPACK_PACKAGE_HOMEPAGE_URL "https://www.sofa-framework.org")
    set(CPACK_PACKAGE_FILE_NAME "SOFA_v${CPACK_PACKAGE_VERSION}")
    if(WIN32)
        set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/share/icons\\\\SOFA.png")
        set(CPACK_PACKAGE_INSTALL_DIRECTORY "SOFA\\\\v${CPACK_PACKAGE_VERSION}")
        if(CMAKE_EXE_LINKER_FLAGS MATCHES ".*machine:x64")
            set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_Win64")
        else() # x86
            set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_Win32")
        endif()
    elseif(UNIX)
        set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/share/icons/SOFA.png")
        set(CPACK_PACKAGE_INSTALL_DIRECTORY "SOFA/v${CPACK_PACKAGE_VERSION}")
        if(APPLE)
            set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_MacOS")
        else()
            set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_Linux")
        endif()
    endif()
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE-LGPL.md")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
    set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_SOURCE_DIR}/README.md")
    ################

    ##################
    # CPack components
    cpack_add_component_group(runtime
        DISPLAY_NAME "Runtime"
        DESCRIPTION "Components needed to run a simulation with ${CPACK_PACKAGE_NAME}"
        EXPANDED
        )
    cpack_add_component_group(development
        DISPLAY_NAME "Development"
        DESCRIPTION "Components needed to write code based on ${CPACK_PACKAGE_NAME}"
        EXPANDED
        )

    cpack_add_component(applications
        DISPLAY_NAME "runSofa Application"
        GROUP runtime
        )
    cpack_add_component(headers
        DISPLAY_NAME "C++ Headers"
        GROUP development
        )
    cpack_add_component(GTest_headers
        DISPLAY_NAME "GTest Headers"
        GROUP development
        )
    cpack_add_component(libraries
        DISPLAY_NAME "Libraries"
        GROUP runtime
        )
    cpack_add_component(resources
        DISPLAY_NAME "Resources"
        GROUP runtime
        )

    set(CPACK_COMPONENTS_ALL applications headers GTest_headers libraries resources)

    set(CPACK_COMPONENT_APPLICATIONS_DISPLAY_NAME "runSofa Application")
    set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "C++ Headers")
    set(CPACK_COMPONENT_GTEST_HEADERS_DISPLAY_NAME "GTest Headers")
    set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
    set(CPACK_COMPONENT_RESOURCES_DISPLAY_NAME "Resources")

    set(CPACK_COMPONENT_APPLICATIONS_GROUP "Runtime")
    set(CPACK_COMPONENT_HEADERS_GROUP "Development")
    set(CPACK_COMPONENT_GTEST_HEADERS_GROUP "Development")
    set(CPACK_COMPONENT_LIBRARIES_GROUP "Runtime")
    set(CPACK_COMPONENT_RESOURCES_GROUP "Runtime")
    ##################

    ######################
    # IFW Generator config
    if(CPACK_BINARY_IFW)
        set(CPACK_IFW_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
        set(CPACK_IFW_PACKAGE_TITLE "${CPACK_PACKAGE_NAME}")
        set(CPACK_IFW_PRODUCT_URL "${CPACK_PACKAGE_HOMEPAGE_URL}")
        set(CPACK_IFW_PACKAGE_LOGO "${CPACK_PACKAGE_ICON}")
        set(CPACK_IFW_TARGET_DIRECTORY "@HomeDir@/${CPACK_PACKAGE_INSTALL_DIRECTORY}")
        if(WIN32)
             set(CPACK_IFW_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/applications/projects/runSofa/runSofa.ico")
        elseif(APPLE)
             set(CPACK_IFW_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/applications/projects/runSofa/runSofa.icns")
        endif()

        cpack_ifw_configure_component_group(runtime
            SORTING_PRIORITY 50
            DEFAULT TRUE
            EXPANDED
            SCRIPT "${CMAKE_SOURCE_DIR}/scripts/qtifw/install.qs"
            )
        cpack_ifw_configure_component_group(development
            SORTING_PRIORITY 10
            DEFAULT TRUE
            EXPANDED
            )

        cpack_ifw_configure_component(applications
            DISPLAY_NAME "runSofa Application"
            DEPENDS runtime
            )
        cpack_ifw_configure_component(headers
            DISPLAY_NAME "C++ Headers"
            DEPENDS development
            )
        cpack_ifw_configure_component(GTest_headers
            DISPLAY_NAME "GTest Headers"
            DEPENDS development
            )
        cpack_ifw_configure_component(libraries
            DISPLAY_NAME "Libraries"
            DEPENDS development
            )
        cpack_ifw_configure_component(resources
            DISPLAY_NAME "Resources"
            DEPENDS runtime
            )
    endif()
    ######################

    #######################
    # NSIS Generator config
    if(CPACK_BINARY_NSIS)
        # There is a bug in NSIS that does not handle full unix paths properly. Make
        # sure there is at least one set of four (4) backslashes.
        set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/applications/projects/runSofa/runSofa.ico")
        set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\runSofa.exe")
        set(CPACK_NSIS_DISPLAY_NAME ${CPACK_PACKAGE_NAME})
        set(CPACK_NSIS_PACKAGE_NAME ${CPACK_PACKAGE_NAME})
        set(CPACK_NSIS_HELP_LINK ${CPACK_PACKAGE_HOMEPAGE_URL})
        set(CPACK_NSIS_URL_INFO_ABOUT ${CPACK_PACKAGE_HOMEPAGE_URL})
        set(CPACK_NSIS_MODIFY_PATH ON)
        set(CPACK_NSIS_CONTACT "contact@sofa-framework.org")
        set(CPACK_NSIS_INSTALL_ROOT "$PROFILE")
        string(CONCAT EXTRA_INSTALL_COMMAND "ExecShell \\\"open\\\" \\\"https://www.sofa-framework.org/thank-you?sofa=" ${CPACK_PACKAGE_VERSION} "&os=windows\\\"")
        set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "${EXTRA_INSTALL_COMMAND}")
        set(CPACK_NSIS_MENU_LINKS
            "https://www.sofa-framework.org/documentation/" "SOFA User Documentation"
            "https://www.sofa-framework.org/api/" "SOFA Developer API"
            "https://github.com/sofa-framework/sofa/discussions" "SOFA Support")
    endif()
    #######################

    if(APPLE AND SOFA_BUILD_APP_BUNDLE)
        set(CPACK_BINARY_DRAGNDROP ON)
        set(CPACK_GENERATOR DragNDrop)

        # Monolithic install containing everything
        unset(CPACK_COMPONENTS_ALL)
        set(CPACK_MONOLITHIC_INSTALL ON)

        # Force CPack install dir to $ENV{DESTDIR}/runSofa.app/Contents/MacOS
        # CPack always adds "$ENV{DESTDIR}" before CPACK_INSTALL_PREFIX
        set(CPACK_SET_DESTDIR TRUE)
        set(CPACK_INSTALL_PREFIX "/runSofa.app/Contents/MacOS")
    endif()

    configure_file(${CMAKE_SOURCE_DIR}/tools/postinstall-fixup/generate-stubfiles.cmake postinstall-scripts/generate-stubfiles.cmake @ONLY)
    set(CPACK_PRE_BUILD_SCRIPTS ${CMAKE_BINARY_DIR}/postinstall-scripts/generate-stubfiles.cmake)

    include(CPack)
endif()
