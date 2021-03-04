include_guard(GLOBAL)
include(CMakePackageConfigHelpers)
include(CMakeParseLibraryList)


# - Create an imported target from a library path and an include dir path.
#   Handle the special case where LIBRARY_PATH is in fact an existing target.
#   Handle the case where LIBRARY_PATH contains the following syntax supported by cmake:
#                                      "optimized /usr/lib/foo.lib debug /usr/lib/foo_d.lib"
#
# sofa_create_imported_target(TARGETNAME LIBRARY_PATH INCLUDE_DIRS)
#  TARGETNAME_Target  - (output) variable which contains the name of the created target.
#                       It is usually contains TARGETNAME with one notable exception.
#                       If LIBRARY_PATH is an existing target, TARGETNAME_Target
#                       contains LIBRARY_PATH instead.
#  TARGETNAME         - (input) the name of the target to create.
#  NAMESPACE          - (input) the namespace where the target is put.
#  LIBRARY_PATH       - (input) the path to the library ( .so or .lib depending on the platform)
#  INCLUDE_DIRS       - (input) include directories associated with the library,
#                       which are added as INTERFACE_INCLUDE_DIRECTORIES for the target.
#
# The typical usage scenario is to convert the absolute paths to a system library that cmake return
# after a find_package call into an imported target. By using the cmake target mechanism, it is
# easier to redistribute a software that depends on system libraries, whose locations are not
# known before hand on the consumer system.
#
# For further reference about this subject :
# http://public.kitware.com/pipermail/cmake-developers/2014-March/009983.html
# Quoted from https://github.com/Kitware/CMake/blob/master/Help/manual/cmake-packages.7.rst
# "Note that it is not advisable to populate any properties which may contain paths,
#  such as :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`,
#  with paths relevnt to dependencies. That would hard-code into installed packages the
#  include directory or library paths for dependencies as found on the machine the package
#  was made on."
#
# Example:
#
# add_library( SHARED myLib )
# find_package(PNG REQUIRED)
# sofa_create_target( PNG MyNamespace "${PNG_LIBRARY}" "${PNG_INCLUDE_DIRS}" )
# target_link_libraries( myLib PUBLIC ${PNG_Target} )
#
macro(sofa_create_target TARGETNAME NAMESPACE LIBRARY_PATH INCLUDE_DIRS)
    # message("TARGETNAME ${TARGETNAME}")
    set(NAMESPACE_TARGETNAME "${NAMESPACE}::${TARGETNAME}")
    # message("LIBRARY_PATH ${LIBRARY_PATH}")
    parse_library_list( "${LIBRARY_PATH}" FOUND LIB_FOUND DEBUG LIB_DEBUG OPT LIB_OPT GENERAL LIB_GEN )

    # message("FOUND ${LIB_FOUND} DEBUG: ${LIB_DEBUG} OPT: ${LIB_OPT} GEN: ${LIB_GEN}")
    if(LIB_FOUND)
        if(NOT TARGET ${TARGETNAME} )
            set(${TARGETNAME}_Target ${NAMESPACE_TARGETNAME} )
            if(NOT TARGET ${NAMESPACE_TARGETNAME} )
                add_library( ${NAMESPACE_TARGETNAME} UNKNOWN IMPORTED )
                set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}" )
                if( NOT ${LIB_DEBUG} STREQUAL "")
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION_DEBUG "${LIB_DEBUG}" )
                endif()
                if( NOT ${LIB_OPT} STREQUAL "")
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIB_OPT}" )
                elseif( NOT ${LIB_GEN} STREQUAL "" )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIB_GEN}" )
                endif()
            endif()
        else()
            message( SEND_ERROR "sofa_create_target error. ${TARGETNAME} is an already an existing TARGET.\
                                 Choose a different name.")
        endif()
    else()
        if(NOT TARGET "${LIBRARY_PATH}" )
            # message("${LIBRARY_PATH} is not a TARGET")
            if(NOT TARGET ${TARGETNAME} )
                # message("${TARGETNAME} is not a TARGET")
                set(${TARGETNAME}_Target ${NAMESPACE_TARGETNAME} )
                if(NOT TARGET ${NAMESPACE_TARGETNAME} )
                    # message("${NAMESPACE_TARGETNAME} is not a TARGET")
                    add_library( ${NAMESPACE_TARGETNAME} UNKNOWN IMPORTED )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES IMPORTED_LOCATION "${LIBRARY_PATH}" )
                    set_target_properties( ${NAMESPACE_TARGETNAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}" )
                endif()
            else()
                message( SEND_ERROR "sofa_create_target error. ${TARGETNAME} is an already an existing TARGET.\
                                     Choose a different name.")
            endif()
        else()
            # message("${LIBRARY_PATH} is a TARGET")
            set(${TARGETNAME}_Target ${LIBRARY_PATH} )
        endif()

    endif()
endmacro()



macro(sofa_add_generic directory name type)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/${directory}" AND IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${directory}")
        string(TOUPPER ${type}_${name} option)
        string(TOLOWER ${type} type_lower)

        # optional parameter to activate/desactivate the option
        #  e.g.  sofa_add_application( path/MYAPP MYAPP APPLICATION ON)
        set(active OFF)
        set(optional_argv3 "${ARGV3}")
        if(optional_argv3)
            set(active ${optional_argv3})
        endif()

        option(${option} "Build the ${name} ${type_lower}." ${active})
        if(${option})
            message("Adding ${type_lower} ${name}")
            add_subdirectory(${directory})
        endif()

        if(TARGET ${name})
            set(target ${name})
            get_target_property(aliased_target ${target} ALIASED_TARGET)
            if(aliased_target)
                set(target ${aliased_target})
            endif()

            set_target_properties(${target} PROPERTIES FOLDER ${type}s) # IDE folder
            set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")

            # Add current target in the internal list only if not present already
            get_property(_allTargets GLOBAL PROPERTY __GlobalTargetList__)
            get_property(_allTargetNames GLOBAL PROPERTY __GlobalTargetNameList__)
            if(NOT ${target} IN_LIST _allTargets)
                set_property(GLOBAL APPEND PROPERTY __GlobalTargetList__ ${target})
            endif()
            if(NOT ${option} IN_LIST _allTargetNames)
                set_property(GLOBAL APPEND PROPERTY __GlobalTargetNameList__ ${option})
            endif()
        endif()
    else()
        message("The ${type_lower} ${name} (${CMAKE_CURRENT_LIST_DIR}/${directory}) does not exist and will be ignored.")
    endif()
endmacro()

macro(sofa_add_collection directory name)
    sofa_add_generic( ${directory} ${name} "Collection" ${ARGV2} )
endmacro()

macro(sofa_add_plugin directory plugin_name)
    sofa_add_generic( ${directory} ${plugin_name} "Plugin" ${ARGV2} )
endmacro()

macro(sofa_add_plugin_experimental directory plugin_name)
    sofa_add_generic( ${directory} ${plugin_name} "Plugin" ${ARGV2} )
    string(TOUPPER "PLUGIN_${plugin_name}" option)
    if(${option})
        message("-- ${plugin_name} is an experimental feature, use it at your own risk.")
    endif()
endmacro()

macro(sofa_add_module directory module_name)
    sofa_add_generic( ${directory} ${module_name} "Module" ${ARGV2} )
endmacro()

macro(sofa_add_module_experimental directory module_name)
    sofa_add_generic( ${directory} ${module_name} "Module" ${ARGV2} )
    string(TOUPPER "MODULE_${module_name}" option)
    if(${option})
        message("-- ${module_name} is an experimental feature, use it at your own risk.")
    endif()
endmacro()

macro(sofa_add_application directory app_name)
    sofa_add_generic( ${directory} ${app_name} "Application" ${ARGV2} )
endmacro()

### External projects management
# Thanks to http://crascit.com/2015/07/25/cmake-gtest/
#
# Use this macro (subdirectory or plugin version) to add out-of-repository projects.
# Usage:
# 1. Add repository configuration in MyProjectDir/ExternalProjectConfig.cmake.in
# 2. Call sofa_add_subdirectory_external(MyProjectDir MyProjectName [ON,OFF] [FETCH_ONLY])
#      or sofa_add_plugin_external(MyProjectDir MyProjectName [ON,OFF] [FETCH_ONLY])
# ON,OFF = execute the fetch by default + enable the fetched plugin (if calling sofa_add_plugin_external)
# FETCH_ONLY = do not "add_subdirectory" the fetched repository
# See plugins/SofaHighOrder for example
#
function(sofa_add_generic_external directory name type)
    set(optionArgs FETCH_ONLY)
    cmake_parse_arguments("ARG" "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Make directory absolute
    if(NOT IS_ABSOLUTE "${directory}")
        set(directory "${CMAKE_CURRENT_LIST_DIR}/${directory}")
    endif()
    if(NOT EXISTS "${directory}")
        message("${directory} does not exist and will be ignored.")
        return()
    endif()

    string(TOLOWER ${type} type_lower)

    # Default value for fetch activation and for plugin activation (if adding a plugin)
    set(active OFF)
    set(optional_argv3 "${ARGV3}")
    if(optional_argv3)
        set(active ${optional_argv3})
    endif()

    # Create option
    string(TOUPPER ${PROJECT_NAME}_FETCH_${name} fetch_enabled)
    option(${fetch_enabled} "Fetch/update ${name} repository." ${active})

    # Setup fetch directory
    set(fetched_dir "${CMAKE_BINARY_DIR}/external_directories/fetched/${name}" )

    # Fetch
    if(${fetch_enabled})
        message("Fetching ${type_lower} ${name}")

        if(NOT EXISTS ${fetched_dir})
            file(MAKE_DIRECTORY "${fetched_dir}/")
        endif()

        # Download and unpack at configure time
        configure_file(${directory}/ExternalProjectConfig.cmake.in ${fetched_dir}/CMakeLists.txt)
        # Copy ExternalProjectConfig.cmake.in in build dir for post-pull recovery in src dir
        file(COPY ${directory}/ExternalProjectConfig.cmake.in DESTINATION ${fetched_dir})

        # Execute commands to fetch content
        message("  Pulling ...")
        file(WRITE "${fetched_dir}/logs.txt" "") # Empty log file
        execute_process(COMMAND "${CMAKE_COMMAND}" -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${fetched_dir}"
            RESULT_VARIABLE generate_exitcode
            OUTPUT_VARIABLE generate_logs ERROR_VARIABLE generate_logs)
        file(APPEND "${fetched_dir}/logs.txt" "${generate_logs}")
        execute_process(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${fetched_dir}"
            RESULT_VARIABLE build_exitcode
            OUTPUT_VARIABLE build_logs ERROR_VARIABLE build_logs)
        file(APPEND "${fetched_dir}/logs.txt" "${build_logs}")

        if(generate_exitcode EQUAL 0 AND build_exitcode EQUAL 0 AND EXISTS "${directory}/.git")
            message("  Sucess.")
            # Add .gitignore for Sofa
            file(WRITE "${directory}/.gitignore" "*")
            # Recover ExternalProjectConfig.cmake.in from build dir (erased by pull)
            file(COPY ${fetched_dir}/ExternalProjectConfig.cmake.in DESTINATION ${directory})
            # Disable fetching for next configure
            set(${fetch_enabled} OFF CACHE BOOL "Fetch/update ${name} repository." FORCE)
            message("  ${fetch_enabled} is now OFF. Set it back to ON to trigger a new fetch.")
        else()
            message(SEND_ERROR "Failed to add external repository ${name}."
                               "\nSee logs in ${fetched_dir}/logs.txt")
        endif()
    endif()

    # Add
    if(EXISTS "${directory}/.git" AND IS_DIRECTORY "${directory}/.git")
        configure_file(${directory}/ExternalProjectConfig.cmake.in ${fetched_dir}/CMakeLists.txt)
        if(NOT ARG_FETCH_ONLY AND "${type}" STREQUAL "Subdirectory")
            add_subdirectory("${directory}")
        elseif(NOT ARG_FETCH_ONLY AND "${type}" STREQUAL "Plugin")
            sofa_add_plugin("${name}" "${name}" ${active})
        endif()
    endif()
endfunction()

function(sofa_add_subdirectory_external directory name)
    sofa_add_generic_external(${directory} ${name} "Subdirectory" ${ARGN})
endfunction()

function(sofa_add_plugin_external directory name)
    sofa_add_generic_external(${directory} ${name} "Plugin" ${ARGN})
endfunction()


# sofa_set_01
#
# Defines a variable to
#   - 1 if VALUE is 1, ON, YES, TRUE, Y, or a non-zero number.
#   - 0 if VALUE is 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, the empty string, or ends in the suffix -NOTFOUND.
# This macro is used to quickly define variables for "#define SOMETHING ${SOMETHING}" in config.h.in files.
# PARENT_SCOPE (option): set the variable only in parent scope
# BOTH_SCOPES (option): set the variable in current AND parent scopes
macro(sofa_set_01 name)
    set(optionArgs PARENT_SCOPE BOTH_SCOPES)
    set(oneValueArgs VALUE)
    set(multiValueArgs)
    cmake_parse_arguments("ARG" "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # Required arguments
    foreach(arg ARG_VALUE)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name)
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()
    if(ARG_VALUE)
        if(ARG_BOTH_SCOPES OR NOT ARG_PARENT_SCOPE)
            set(${name} 1)
        endif()
        if(ARG_BOTH_SCOPES OR ARG_PARENT_SCOPE)
            set(${name} 1 PARENT_SCOPE)
        endif()
    else()
        if(ARG_BOTH_SCOPES OR NOT ARG_PARENT_SCOPE)
            set(${name} 0)
        endif()
        if(ARG_BOTH_SCOPES OR ARG_PARENT_SCOPE)
            set(${name} 0 PARENT_SCOPE)
        endif()
    endif()
endmacro()


# sofa_find_package
#
# Defines a PROJECTNAME_HAVE_PACKAGENAME variable to be used in:
#  - XXXConfig.cmake.in to decide if find_dependency must be done
#  - config.h.in as a #cmakedefine
#  - config.h.in as a #define SOMETHING ${SOMETHING}
# BOTH_SCOPES (option): set the variable in current AND parent scopes
macro(sofa_find_package name)
    set(optionArgs QUIET REQUIRED BOTH_SCOPES)
    set(oneValueArgs)
    set(multiValueArgs COMPONENTS OPTIONAL_COMPONENTS)
    cmake_parse_arguments("ARG" "${optionArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(find_package_args ${ARGN})
    if(find_package_args)
        list(REMOVE_ITEM find_package_args "BOTH_SCOPES")
    endif()

    if(NOT TARGET ${name})
        find_package(${name} ${find_package_args})
    else()
        # Dirty ? set the variable _FOUND if the target is present
        if(NOT ${name}_FOUND)
            set(${name}_FOUND TRUE)
        endif()
    endif()

    string(TOUPPER ${name} name_upper)
    string(TOUPPER ${PROJECT_NAME} project_upper)
    string(REPLACE "." "_" name_upper "${name_upper}")
    string(REPLACE "." "_" project_upper "${project_upper}")

    set(scopes "") # nothing = current scope only
    if(ARG_BOTH_SCOPES)
        set(scopes "BOTH_SCOPES")
    endif()
    if(ARG_COMPONENTS OR ARG_OPTIONAL_COMPONENTS)
        set(all_components_found TRUE)
        foreach(component ${ARG_COMPONENTS} ${ARG_OPTIONAL_COMPONENTS})
            string(TOUPPER ${component} component_upper)
            if(TARGET ${name}::${component})
                sofa_set_01(${project_upper}_HAVE_${name_upper}_${component_upper} VALUE TRUE ${scopes})
            else()
                set(all_components_found FALSE)
                sofa_set_01(${project_upper}_HAVE_${name_upper}_${component_upper} VALUE FALSE ${scopes})
            endif()
        endforeach()
        sofa_set_01(${project_upper}_HAVE_${name_upper} VALUE ${all_components_found} ${scopes})
    else()
        if(${name}_FOUND OR ${name_upper}_FOUND)
            sofa_set_01(${project_upper}_HAVE_${name_upper} VALUE TRUE ${scopes})
        else()
            sofa_set_01(${project_upper}_HAVE_${name_upper} VALUE FALSE ${scopes})
        endif()
    endif()
endmacro()


# sofa_set_targets_release_only
#
# Force dependency targets to link with their Release version
# (even if we are building in Debug mode).
# It eases deps management, especially on Windows with the WinDepPack.
macro(sofa_set_targets_release_only)
    foreach(target ${ARGN})
        if(NOT TARGET ${target})
            message("sofa_set_targets_release_only: ${target} is not a target")
            continue()
        endif()
        set_target_properties(${target} PROPERTIES
            MAP_IMPORTED_CONFIG_MINSIZEREL Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
            MAP_IMPORTED_CONFIG_DEBUG Release
            )
    endforeach()
endmacro()
