cmake_minimum_required(VERSION 2.8.12)

project(Sofa)

# Enable the organisation in folders for generators that support
# it. (E.g. some versions of Visual Studio have 'solution folders')
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    set( CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories." FORCE )
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )

message("Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Set the output directories globally
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)



# Environment
set(SOFA_CMAKE_DIR "${Sofa_SOURCE_DIR}/cmake" )
list(APPEND CMAKE_MODULE_PATH "${SOFA_CMAKE_DIR}/Modules")
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

include(CMakeSofaBuildFlags)



# Sofa libraries
set(SOFA_EXTERNALS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sofa-win-dependencies-21-11-2013")
set(SOFA_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaDeps")
set(SOFA_FRAMEWORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaFramework")
set(SOFA_SIMULATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaSimulation")
set(SOFA_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaBase")
set(SOFA_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaCommon")
set(SOFA_GENERAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaGeneral")
set(SOFA_ADVANCED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaAdvanced")
set(SOFA_MISC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaMisc")
set(SOFA_GUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaGui")
set(SOFA_PLUGINS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SofaPlugins")
set(SOFA_EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/examples")
set(SOFA_SHARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share")

# cmake functions specific to sofa that needs to be redistributed.
add_subdirectory( ${SOFA_CMAKE_DIR} )

# Dependency pack for Windows
if(MSVC)
    if(EXISTS ${SOFA_EXTERNALS_DIR} )
        list(APPEND CMAKE_INCLUDE_PATH ${SOFA_EXTERNALS_DIR}/include)
        if(CMAKE_CL_64)
            list(APPEND CMAKE_LIBRARY_PATH ${SOFA_EXTERNALS_DIR}/lib/win64)
        else()
            list(APPEND CMAKE_LIBRARY_PATH ${SOFA_EXTERNALS_DIR}/lib/win32)
        endif()
        add_subdirectory("${SOFA_EXTERNALS_DIR}")
    endif()
endif()

# Install resource files
install(DIRECTORY "${SOFA_SHARE_DIR}/" DESTINATION share/sofa)
install(DIRECTORY "${SOFA_EXAMPLES_DIR}/" DESTINATION share/sofa/examples)
# Create etc/sofa.ini: it contains the paths to share/ and examples/. In the
# build directory, it points to the source tree, whereas in the install
# directory, it contains to relative paths to the installed resource directory.
set(SHARE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/share)
set(EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)
configure_file(etc/sofa.ini.in ${CMAKE_BINARY_DIR}/etc/sofa.ini)
set(SHARE_DIR ../share/sofa)
set(EXAMPLES_DIR ../share/sofa/examples)
configure_file(etc/sofa.ini.in ${CMAKE_BINARY_DIR}/etc/installedSofa.ini)

install(FILES ${CMAKE_BINARY_DIR}/etc/installedSofa.ini
        DESTINATION etc
        RENAME sofa.ini)

# deps
add_subdirectory("${SOFA_DEPS_DIR}")

# framework
add_subdirectory("${SOFA_FRAMEWORK_DIR}")

# simulation
add_subdirectory("${SOFA_SIMULATION_DIR}")

# base
add_subdirectory("${SOFA_BASE_DIR}")

# common
add_subdirectory("${SOFA_COMMON_DIR}")

# general
add_subdirectory("${SOFA_GENERAL_DIR}")

# advanced
add_subdirectory("${SOFA_ADVANCED_DIR}")

# misc
add_subdirectory("${SOFA_MISC_DIR}")

# gui
add_subdirectory("${SOFA_GUI_DIR}")

# plugins
add_subdirectory("${SOFA_PLUGINS_DIR}")
