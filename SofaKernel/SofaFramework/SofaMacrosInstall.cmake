include_guard(GLOBAL)
include(CMakePackageConfigHelpers)
include(CMakeParseLibraryList)


# sofa_create_package_with_targets(
#     PACKAGE_NAME <package_name>
#     PACKAGE_VERSION <project_version>
#     TARGETS <target1> [<target2>...] [AUTO_SET_TARGET_PROPERTIES]
#     [INCLUDE_SOURCE_DIR <include_source_dir>]
#     [INCLUDE_INSTALL_DIR <include_install_dir>]
#     [RELOCATABLE <install_dir>]
#     )
#
# This is the global macro for creating a plugin in SOFA.
#
# [optional] AUTO_SET_TARGET_PROPERTIES
#   Use AUTO_SET_TARGET_PROPERTIES to enable default properties setting
#   on all targets (see sofa_auto_set_target_properties).
#
# [optional] INCLUDE_SOURCE_DIR <include_source_dir>
#   Directory from which headers will be copied, respecting subdirectories tree.
#
# [optional] INCLUDE_INSTALL_DIR <include_install_dir>
#   Directory in which headers will be copied into <CMAKE_INSTALL_PREFIX>/include/<include_install_dir>
#
# [optional] RELOCATABLE <install_dir>
#   If building through SOFA, package will be integrally installed in <install_dir>
#   instead of being dispatched in SOFA install directory (between bin, libs, share, ...).
#   If not building through SOFA, RELOCATABLE has no effect.
macro(sofa_create_package_with_targets)
    set(oneValueArgs PACKAGE_NAME PACKAGE_VERSION INCLUDE_ROOT_DIR INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR EXAMPLE_INSTALL_DIR RELOCATABLE)
    set(multiValueArgs TARGETS)
    set(optionalArgs AUTO_SET_TARGET_PROPERTIES)
    cmake_parse_arguments("ARG" "${optionalArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # Required arguments
    foreach(arg ARG_PACKAGE_NAME ARG_PACKAGE_VERSION ARG_TARGETS)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name)
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()
    # Default value for INCLUDE_INSTALL_DIR
    set(child_args ${ARGV})
    if(NOT ARG_INCLUDE_INSTALL_DIR)
        list(APPEND child_args INCLUDE_INSTALL_DIR "${ARG_PACKAGE_NAME}")
    endif()

    sofa_create_package(${child_args})
    sofa_add_targets_to_package(${child_args})

    if(ARG_RELOCATABLE)
        sofa_set_target_install_relocatable(${ARG_PACKAGE_NAME} ${ARG_RELOCATABLE})
    endif()
endmacro()


# sofa_create_package(
#     PACKAGE_NAME <package_name>
#     PACKAGE_VERSION <package_version>
#     )
#
# Create CMake package configuration files
# - In the build tree:
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfig.cmake
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfigVersion.cmake
# - In the install tree:
#   - lib/cmake/Foo/FooConfigVersion.cmake
#   - lib/cmake/Foo/FooConfig.cmake
#   - lib/cmake/Foo/FooTargets.cmake
#
# This macro factorizes boilerplate CMake code for the different
# packages in Sofa.  It assumes that there is a FooConfig.cmake.in
# file template in the same directory.  For example, if a package Foo
# depends on Bar and Baz, and creates the targets Foo and Qux, here is
# a typical FooConfig.cmake.in:
#
# @PACKAGE_INIT@
#
# find_package(Bar QUIET REQUIRED)
# find_package(Baz QUIET REQUIRED)
#
# if(NOT TARGET Qux)
#     include("${CMAKE_CURRENT_LIST_DIR}/FooTargets.cmake")
# endif()
#
# check_required_components(Foo Qux)
macro(sofa_create_package)
    set(oneValueArgs PACKAGE_NAME PACKAGE_VERSION INCLUDE_ROOT_DIR INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR EXAMPLE_INSTALL_DIR RELOCATABLE)
    set(multiValueArgs TARGETS)
    set(optionalArgs AUTO_SET_TARGET_PROPERTIES)
    cmake_parse_arguments("ARG" "${optionalArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # Required arguments
    foreach(arg ARG_PACKAGE_NAME ARG_PACKAGE_VERSION)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name)
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()

    ## <package_name>Targets.cmake
    install(EXPORT ${ARG_PACKAGE_NAME}Targets DESTINATION "lib/cmake/${ARG_PACKAGE_NAME}" COMPONENT headers)

    ## <package_name>ConfigVersion.cmake
    set(filename ${ARG_PACKAGE_NAME}ConfigVersion.cmake)
    write_basic_package_version_file(${filename} VERSION ${ARG_PACKAGE_VERSION} COMPATIBILITY ExactVersion)
    set(PACKAGE_GUARD "include_guard()")
    configure_file("${CMAKE_CURRENT_BINARY_DIR}/${filename}" "${CMAKE_BINARY_DIR}/cmake/${filename}" COPYONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${filename}" DESTINATION "lib/cmake/${ARG_PACKAGE_NAME}" COMPONENT headers)

    ### <package_name>Config.cmake
    configure_package_config_file(
        ${ARG_PACKAGE_NAME}Config.cmake.in
        "${CMAKE_BINARY_DIR}/cmake/${ARG_PACKAGE_NAME}Config.cmake"
        INSTALL_DESTINATION "lib/cmake/${ARG_PACKAGE_NAME}"
        )
    install(FILES "${CMAKE_BINARY_DIR}/cmake/${ARG_PACKAGE_NAME}Config.cmake" DESTINATION "lib/cmake/${ARG_PACKAGE_NAME}" COMPONENT headers)
endmacro()


# sofa_add_targets_to_package(
#     PACKAGE_NAME <package_name>
#     TARGETS <target1> [<target2>...] [AUTO_SET_TARGET_PROPERTIES]
#     )
#
# This macro adds targets to an existing package (created with
# sofa_create_package_with_targets or sofa_create_package).
#
# [optional] AUTO_SET_TARGET_PROPERTIES
#   Use AUTO_SET_TARGET_PROPERTIES to enable default properties setting
#   on all targets (see sofa_auto_set_target_properties).
macro(sofa_add_targets_to_package)
    set(oneValueArgs PACKAGE_NAME PACKAGE_VERSION INCLUDE_ROOT_DIR INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR EXAMPLE_INSTALL_DIR RELOCATABLE)
    set(multiValueArgs TARGETS)
    set(optionalArgs AUTO_SET_TARGET_PROPERTIES)
    cmake_parse_arguments("ARG" "${optionalArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # Required arguments
    foreach(arg ARG_PACKAGE_NAME ARG_TARGETS)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name)
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()
    # Default value for INCLUDE_INSTALL_DIR
    set(child_args ${ARGV})
    if(NOT ARG_INCLUDE_INSTALL_DIR)
        list(APPEND child_args INCLUDE_INSTALL_DIR "${ARG_PACKAGE_NAME}/${PROJECT_NAME}")
    endif()

    if(ARG_AUTO_SET_TARGET_PROPERTIES)
        sofa_auto_set_target_properties(${child_args})
    endif()

    sofa_install_targets_in_package(${child_args})
endmacro()

# sofa_auto_set_target_properties(
#     PACKAGE_NAME <package_name>
#     TARGETS <target1> [<target2>...]
#     [INCLUDE_INSTALL_DIR <include_install_dir>]
#     )
#
# Auto set these properties on given targets:
# - DEBUG_POSTFIX: "_d"
# - VERSION: ${target}_VERSION if possible, Sofa_VERSION otherwise
# - COMPILE_DEFINITIONS: for targets like SofaModuleName, add
#     "-DSOFA_BUILD_MODULE_NAME" (old style) and "-DSOFA_BUILD_SOFAMODULENAME" (new style)
# - INCLUDE_DIRECTORIES: if not already set, add as PUBLIC include dirs
#     2 BUILD_INTERFACE (source dir and build dir) and 1 INSTALL_INTERFACE (install dir)
macro(sofa_auto_set_target_properties)
    set(oneValueArgs PACKAGE_NAME PACKAGE_VERSION INCLUDE_ROOT_DIR INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR EXAMPLE_INSTALL_DIR RELOCATABLE)
    set(multiValueArgs TARGETS)
    set(optionalArgs AUTO_SET_TARGET_PROPERTIES)
    cmake_parse_arguments("ARG" "${optionalArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # Required arguments
    foreach(arg ARG_PACKAGE_NAME ARG_TARGETS ARG_INCLUDE_INSTALL_DIR)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name)
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()
    
    string(TOUPPER "${ARG_PACKAGE_NAME}" SOFA_PACKAGE_NAME_UPPER)

    foreach(target ${ARG_TARGETS}) # Most of the time there is only one target
        # Handle eventual alias
        get_target_property(aliased_target ${target} ALIASED_TARGET)
        if(aliased_target)
            set(target ${aliased_target})
        endif()

        string(TOUPPER "${target}" sofa_target_name_upper)
        set(${sofa_target_name_upper}_TARGET "${sofa_target_name_upper}")

        # Set target properties
        if(NOT "${target}" STREQUAL "${ARG_PACKAGE_NAME}") # Target is inside a package
            set_target_properties(${target} PROPERTIES FOLDER ${ARG_PACKAGE_NAME}) # IDE folder
        endif()
        set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
        set(version "")
        if(${target}_VERSION VERSION_GREATER "0.0")
            set(version ${${target}_VERSION})
        elseif(ARG_PACKAGE_VERSION VERSION_GREATER "0.0")
            set(version ${ARG_PACKAGE_VERSION})
        elseif(Sofa_VERSION VERSION_GREATER "0.0")
            # Default to Sofa_VERSION for all SOFA modules
            set(version ${Sofa_VERSION})
        endif()
        set_target_properties(${target} PROPERTIES VERSION "${version}")
        set(${sofa_target_name_upper}_VERSION "${version}")
        set(PROJECT_VERSION "${version}") # warning: dangerous to touch this variable?

        if(target MATCHES "^Sofa")
            # TODO: Deprecate this backward compatibility and replace all the macros
            # with old style: SofaModuleName -> SOFA_BUILD_MODULE_NAME + SOFA_MODULE_NAME_API
            # by new style: SofaModuleName -> SOFA_BUILD_SOFAMODULENAME + SOFA_SOFAMODULENAME_API
            string(REGEX REPLACE "([^A-Z])([A-Z])" "\\1_\\2" sofa_target_oldname "${target}")
            string(REPLACE "Sofa" "" sofa_target_oldname "${sofa_target_oldname}")
            string(TOUPPER "${sofa_target_oldname}" sofa_target_oldname_upper)
            target_compile_definitions(${target} PRIVATE "-DSOFA_BUILD${sofa_target_oldname_upper}")
        endif()
        target_compile_definitions(${target} PRIVATE "-DSOFA_BUILD_${sofa_target_name_upper}")

        # Set target include directories (if not already set manually)
        set(include_source_root "${CMAKE_CURRENT_SOURCE_DIR}/..") # default but bad practice
        if(ARG_INCLUDE_SOURCE_DIR)
            if(IS_ABSOLUTE "${ARG_INCLUDE_SOURCE_DIR}")
                set(include_source_root "${ARG_INCLUDE_SOURCE_DIR}")
            else()
                set(include_source_root "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INCLUDE_SOURCE_DIR}")
            endif()
        endif()
        get_target_property(target_include_dirs ${target} "INCLUDE_DIRECTORIES")
        
        if(NOT "\$<BUILD_INTERFACE:${include_source_root}>" IN_LIST target_include_dirs)
            target_include_directories(${target} PUBLIC "$<BUILD_INTERFACE:${include_source_root}>")
        endif()
        if(NOT "\$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${ARG_PACKAGE_NAME}>" IN_LIST target_include_dirs)
            target_include_directories(${target} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${ARG_PACKAGE_NAME}>")
        endif()
        
        if(ARG_RELOCATABLE)
            if(NOT "\$<INSTALL_INTERFACE:include>" IN_LIST target_include_dirs)
                target_include_directories(${target} PUBLIC "$<INSTALL_INTERFACE:include>")
            endif()
        elseif("${ARG_INCLUDE_INSTALL_DIR}" MATCHES "^${ARG_PACKAGE_NAME}")
            if(NOT "\$<INSTALL_INTERFACE:include/${ARG_PACKAGE_NAME}>" IN_LIST target_include_dirs)
                target_include_directories(${target} PUBLIC "$<INSTALL_INTERFACE:include/${ARG_PACKAGE_NAME}>")
            endif()
        else()
            if(NOT "\$<INSTALL_INTERFACE:include/${ARG_INCLUDE_INSTALL_DIR}>" IN_LIST target_include_dirs)
                target_include_directories(${target} PUBLIC "$<INSTALL_INTERFACE:include/${ARG_INCLUDE_INSTALL_DIR}>")
            endif()
        endif()
        #get_target_property(target_include_dirs ${target} "INCLUDE_DIRECTORIES")
        #message("${ARG_PACKAGE_NAME}: target_include_dirs = ${target_include_dirs}")

        set_target_properties(${target} PROPERTIES RELOCATABLE_INSTALL_DIR "${ARG_RELOCATABLE}/${ARG_PACKAGE_NAME}")

        get_target_property(target_deps ${target} "LINK_LIBRARIES")
        get_target_property(target_rpath ${target} "INSTALL_RPATH")
        foreach(dep ${target_deps})
            if(TARGET ${dep})
                get_target_property(aliased_dep ${dep} ALIASED_TARGET)
                if(aliased_dep)
                    set(dep ${aliased_dep})
                endif()
                get_target_property(dep_type ${dep} TYPE)
                if("${dep_type}" STREQUAL "SHARED_LIBRARY")
                    get_target_property(dep_reloc_install_dir ${dep} "RELOCATABLE_INSTALL_DIR")
                    if(dep_reloc_install_dir)
                        # the dependency is relocatable
                        if(ARG_RELOCATABLE) 
                            # current target is relocatable
                            list(APPEND target_rpath
                                "$ORIGIN/../../../${dep_reloc_install_dir}/lib"
                                "$$ORIGIN/../../../${dep_reloc_install_dir}/lib"
                                )
                        else()
                            # current target is NOT relocatable
                            list(APPEND target_rpath
                                "$ORIGIN/../${dep_reloc_install_dir}/lib"
                                "$$ORIGIN/../${dep_reloc_install_dir}/lib"
                                )
                        endif()
                    else()
                        # the dependency is NOT relocatable
                        if(ARG_RELOCATABLE)
                            # current target is relocatable
                            list(APPEND target_rpath
                                "$ORIGIN/../../../lib"
                                "$$ORIGIN/../../../lib"
                                )
                        endif()
                    endif()
                endif()
            endif()
        endforeach()
        list(REMOVE_DUPLICATES target_rpath)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "${target_rpath}")

    endforeach()
endmacro()


# sofa_install_targets_in_package(
#     PACKAGE_NAME <package_name>
#     TARGETS <target1> [<target2>...] [AUTO_SET_TARGET_PROPERTIES]
#     [INCLUDE_INSTALL_DIR <include_install_dir>]
#     )
#
# Export given targets and install binaries, libraries, headers, examples, resources
#
# INCLUDE_INSTALL_DIR <include_install_dir>
#   Directory in which headers will be copied into <CMAKE_INSTALL_PREFIX>/include/<include_install_dir>
macro(sofa_install_targets_in_package)
    set(oneValueArgs PACKAGE_NAME PACKAGE_VERSION INCLUDE_ROOT_DIR INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR EXAMPLE_INSTALL_DIR RELOCATABLE)
    set(multiValueArgs TARGETS)
    set(optionalArgs AUTO_SET_TARGET_PROPERTIES)
    cmake_parse_arguments("ARG" "${optionalArgs}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # Required arguments
    foreach(arg ARG_PACKAGE_NAME ARG_TARGETS ARG_INCLUDE_INSTALL_DIR)
        if("${${arg}}" STREQUAL "")
            string(SUBSTRING "${arg}" 4 -1 arg_name)
            message(SEND_ERROR "Missing parameter ${arg_name}.")
        endif()
    endforeach()

    install(TARGETS ${ARG_TARGETS}
            EXPORT ${ARG_PACKAGE_NAME}Targets
            RUNTIME DESTINATION "bin" COMPONENT applications
            LIBRARY DESTINATION "lib" COMPONENT libraries
            ARCHIVE DESTINATION "lib" COMPONENT libraries
            PUBLIC_HEADER DESTINATION "include/${ARG_INCLUDE_INSTALL_DIR}" COMPONENT headers

            # [MacOS] install runSofa above the already populated runSofa.app (see CMAKE_INSTALL_PREFIX)
            BUNDLE DESTINATION "../../.." COMPONENT applications
            )

    set(include_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    if(ARG_INCLUDE_SOURCE_DIR)
        if(IS_ABSOLUTE "${ARG_INCLUDE_SOURCE_DIR}")
            set(include_source_dir "${ARG_INCLUDE_SOURCE_DIR}")
        else()
            set(include_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_INCLUDE_SOURCE_DIR}")
        endif()
    endif()

    set(example_install_dir "share/sofa/examples/${ARG_PACKAGE_NAME}")
    if(ARG_EXAMPLE_INSTALL_DIR)
        set(example_install_dir "${ARG_EXAMPLE_INSTALL_DIR}")
    endif()

    string(TOUPPER "${ARG_PACKAGE_NAME}" SOFA_PACKAGE_NAME_UPPER)

    foreach(target ${ARG_TARGETS}) # Most of the time there is only one target
        # Configure and install headers
        get_target_property(target_sources ${target} SOURCES)
        list(FILTER target_sources INCLUDE REGEX ".*(\\.h\\.in|\\.h|\\.inl)$") # keep only headers
        foreach(header_file ${target_sources})
            if(NOT IS_ABSOLUTE "${header_file}")
                set(header_file "${CMAKE_CURRENT_SOURCE_DIR}/${header_file}")
            endif()
            if("${header_file}" MATCHES "${CMAKE_CURRENT_BINARY_DIR}/.*")
                file(RELATIVE_PATH header_relative_path "${CMAKE_CURRENT_BINARY_DIR}" "${header_file}")
            else()
                file(RELATIVE_PATH header_relative_path "${include_source_dir}" "${header_file}")
            endif()
            get_filename_component(header_relative_dir ${header_relative_path} DIRECTORY)
            get_filename_component(header_filename ${header_file} NAME_WE)

            # Optimize build dir
            set(header_relative_dir_for_build "${header_relative_dir}")
            string(REPLACE "../" "" header_relative_dir_for_build "${header_relative_dir_for_build}") # keep out-of-tree headers
            if("${target}" STREQUAL "${ARG_PACKAGE_NAME}") # Target is a package
                if("${header_relative_dir_for_build}" STREQUAL "") # Headers are not in a subdirectory
                    set(header_relative_dir_for_build "${target}")
                endif()
                if(NOT "${target}" STREQUAL "SofaFramework" AND
                   NOT "${ARG_INCLUDE_INSTALL_DIR}/${header_relative_dir_for_build}" MATCHES "${target}/${target}")
                    # Force include/PackageName/PackageName/... layout for package headers in build directory
                    set(header_relative_dir_for_build "${target}/${header_relative_dir_for_build}")
                endif()
            endif()

            # Finalize dirs
            if(ARG_RELOCATABLE)
                set(header_install_dir "include/${header_relative_dir_for_build}")
            else()
                # headers install-dir tree = headers build-dir tree
                set(header_install_dir "include/${ARG_INCLUDE_INSTALL_DIR}/${header_relative_dir_for_build}")
            endif()
            file(TO_CMAKE_PATH "${header_install_dir}" header_install_dir)

            # Configure and install
            get_target_property(public_header ${target} PUBLIC_HEADER)
            if(header_file MATCHES ".*\\.h\\.in$")
                # header to configure and install
                file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/include/${ARG_INCLUDE_INSTALL_DIR}/${header_relative_dir_for_build}/${header_filename}.h" configured_file)
                configure_file("${header_file}" "${configured_file}")
                install(FILES "${configured_file}" DESTINATION "${header_install_dir}" COMPONENT headers)
                #message("${ARG_PACKAGE_NAME}: configured_file = ${configured_file}")
                #message("${ARG_PACKAGE_NAME}: configured_install_dir = ${header_install_dir}")
            elseif("${public_header}" STREQUAL "public_header-NOTFOUND" AND NOT "${ARG_INCLUDE_INSTALL_DIR}" STREQUAL "")
                # header to install
                install(FILES ${header_file} DESTINATION "${header_install_dir}" COMPONENT headers)
                #message("${ARG_PACKAGE_NAME}: header_file = ${header_file}")
                #message("${ARG_PACKAGE_NAME}: header_install_dir = ${header_install_dir}\n")
            endif()
        endforeach()
    endforeach()

    # Install examples and scenes
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/"
                DESTINATION "${example_install_dir}"
                COMPONENT resources)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scenes")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scenes/"
                DESTINATION "${example_install_dir}"
                COMPONENT resources)
    endif()

    # Install info files (README, license, etc.)
    file(GLOB txt_files "*.txt" RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} LIST_DIRECTORIES false)
    list(FILTER txt_files EXCLUDE REGEX "CMakeLists.txt")
    file(GLOB md_files "*.md" RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} LIST_DIRECTORIES false)
    if(ARG_RELOCATABLE)
        set(info_install_dir ".")
    else()
        set(info_install_dir "include/${ARG_INCLUDE_INSTALL_DIR}")
    endif()
    install(FILES ${txt_files} ${md_files} DESTINATION "${info_install_dir}" COMPONENT headers)
endmacro()


# sofa_set_target_install_relocatable
#   TARGET MUST EXIST, TO BE CALLED AFTER add_library
# Content:
#   If building out of SOFA: does nothing.
#   If building through SOFA: call add_custom_target with custom commands to obtain a self-contained relocatable install.
#   Self-contained plugins are useful to build modular binaries: they do not "pollute" SOFA install
#   with self-contained plugins SOFA install will always look the same, no matter how many plugins are included.
# Effect:
#   add_custom_target will add the line 'set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${install_dir}/${name}")' at the top of the
#   plugin's cmake_install.cmake to force the plugin to be installed in it's own directory instead of in SOFA's install directory
#   (look at the build directory of any plugin to find an example of cmake_install.cmake).
function(sofa_set_target_install_relocatable target install_dir)
    if(NOT "${CMAKE_PROJECT_NAME}" STREQUAL "Sofa")
        # not building through SOFA
        return()
    endif()
    if(NOT TARGET ${target})
        message(WARNING "sofa_set_target_install_relocatable: \"${target}\" is not an existing target.")
        return()
    endif()

    get_target_property(target_binary_dir ${target} BINARY_DIR)

    # Remove cmakepatch file at each configure
    file(REMOVE "${target_binary_dir}/cmake_install.cmakepatch")

    # Hack to make installed plugin independant and keep the add_subdirectory mechanism
    # Does not fail if cmakepatch file already exists thanks to "|| true"
    if(WIN32)
        set(escaped_dollar "\$\$")
        if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (Visual Studio)
            set(escaped_dollar "\$")
        endif()
        string(REGEX REPLACE "/" "\\\\" target_binary_dir_windows "${target_binary_dir}")
        add_custom_target(${target}_relocatable_install ALL
            COMMENT "${target}: Patching cmake_install.cmake"
            COMMAND
                if not exist \"${target_binary_dir}/cmake_install.cmakepatch\"
                echo set ( CMAKE_INSTALL_PREFIX_BACK \"${escaped_dollar}\{CMAKE_INSTALL_PREFIX\}\" )
                    > \"${target_binary_dir}/cmake_install.cmakepatch\"
                && echo set ( CMAKE_INSTALL_PREFIX \"${escaped_dollar}\{CMAKE_INSTALL_PREFIX\}/${install_dir}/${target}\" )
                    >> \"${target_binary_dir}/cmake_install.cmakepatch\"
                && type \"${target_binary_dir_windows}\\cmake_install.cmake\" >> \"${target_binary_dir_windows}\\cmake_install.cmakepatch\"
                && echo set ( CMAKE_INSTALL_PREFIX \"${escaped_dollar}\{CMAKE_INSTALL_PREFIX_BACK\}\" )
                    >> \"${target_binary_dir}/cmake_install.cmakepatch\"
                && ${CMAKE_COMMAND} -E copy \"${target_binary_dir}/cmake_install.cmakepatch\" \"${target_binary_dir}/cmake_install.cmake\"
            )
    else()
        add_custom_target(${target}_relocatable_install ALL
            COMMENT "${target}: Patching cmake_install.cmake"
            COMMAND
                test ! -e ${target_binary_dir}/cmake_install.cmakepatch
                && echo \" set ( CMAKE_INSTALL_PREFIX_BACK \\"\\$$\{CMAKE_INSTALL_PREFIX\}\\" ) \"
                    > "${target_binary_dir}/cmake_install.cmakepatch"
                && echo \" set ( CMAKE_INSTALL_PREFIX \\"\\$$\{CMAKE_INSTALL_PREFIX\}/${install_dir}/${target}\\" ) \"
                    >> "${target_binary_dir}/cmake_install.cmakepatch"
                && cat ${target_binary_dir}/cmake_install.cmake >> ${target_binary_dir}/cmake_install.cmakepatch
                && echo \" set ( CMAKE_INSTALL_PREFIX \\"\\$$\{CMAKE_INSTALL_PREFIX_BACK\}\\" ) \"
                    >> "${target_binary_dir}/cmake_install.cmakepatch"
                && ${CMAKE_COMMAND} -E copy ${target_binary_dir}/cmake_install.cmakepatch ${target_binary_dir}/cmake_install.cmake
                || true
            )
    endif()
    set_target_properties(${target}_relocatable_install PROPERTIES FOLDER "relocatable_install")
endfunction()


# Get path of all library versions (involving symbolic links) for a specified library
function(sofa_install_libraries)
    set(options NO_COPY)
    set(multiValueArgs TARGETS LIBRARIES PATHS)
    cmake_parse_arguments("sofa_install_libraries" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(no_copy ${sofa_install_libraries_NO_COPY})
    set(targets ${sofa_install_libraries_TARGETS})
    set(lib_paths ${sofa_install_libraries_PATHS})
    if(sofa_install_libraries_LIBRARIES)
        message(WARNING "sofa_install_libraries: LIBRARIES parameter is deprecated, use PATHS instead.")
        list(APPEND lib_paths "${sofa_install_libraries_LIBRARIES}")
    endif()

    foreach(target ${targets})
        get_target_property(target_location ${target} LOCATION_${CMAKE_BUILD_TYPE})
        get_target_property(is_framework ${target} FRAMEWORK)
        if(APPLE AND is_framework)
            get_filename_component(target_location ${target_location} DIRECTORY) # parent dir
            install(DIRECTORY ${target_location} DESTINATION "lib" COMPONENT applications)
        else()
            list(APPEND lib_paths "${target_location}")
        endif()
    endforeach()

    if(lib_paths)
        PARSE_LIBRARY_LIST(${lib_paths}
            FOUND   parseOk
            DEBUG   LIBRARIES_DEBUG
            OPT     LIBRARIES_RELEASE
            GENERAL LIBRARIES_GENERAL)

        if(parseOk)
            if(CMAKE_BUILD_TYPE MATCHES DEBUG)
                set(lib_paths ${LIBRARIES_DEBUG})
            else()
                set(lib_paths ${LIBRARIES_RELEASE})
            endif()
        endif()
    else()
        message(WARNING "sofa_install_libraries: no lib found with ${ARGV}")
    endif()

    foreach(lib_path ${lib_paths})
        if(EXISTS ${lib_path})
            get_filename_component(LIBREAL ${lib_path} REALPATH)
            get_filename_component(LIBREAL_NAME ${LIBREAL} NAME_WE)
            get_filename_component(LIBREAL_PATH ${LIBREAL} PATH)

            # In "${LIBREAL_NAME}." the dot is a real dot, not a regex symbol
            # CMAKE_*_LIBRARY_SUFFIX also start with a dot
            # So regex is:
            # <lib_path> <slash> <library_name> <dot> <dll/so/dylib/...>
            # or:
            # <lib_path> <slash> <library_name> <dot> <anything> <dot> <dll/so/dylib/...>
            file(GLOB_RECURSE SHARED_LIBS
                "${LIBREAL_PATH}/${LIBREAL_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}*" # libtiff.dll
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9][0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}*" # libpng16.dll
                "${LIBREAL_PATH}/${LIBREAL_NAME}.*${CMAKE_SHARED_LIBRARY_SUFFIX}*" # libpng.16.dylib
                )
            file(GLOB_RECURSE STATIC_LIBS
                "${LIBREAL_PATH}/${LIBREAL_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9]${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}[0-9][0-9]${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                "${LIBREAL_PATH}/${LIBREAL_NAME}.*${CMAKE_STATIC_LIBRARY_SUFFIX}*"
                )

            if(WIN32)
                install(FILES ${SHARED_LIBS} DESTINATION "bin" COMPONENT applications)
            else()
                install(FILES ${SHARED_LIBS} DESTINATION "lib" COMPONENT applications)
            endif()
            install(FILES ${STATIC_LIBS} DESTINATION "lib" COMPONENT libraries)
        endif()
    endforeach()

    if(WIN32 AND NOT no_copy)
        sofa_copy_libraries(PATHS ${lib_paths})
    endif()
endfunction()

function(sofa_install_get_libraries library)
    message(WARNING "sofa_install_get_libraries() is deprecated. Please use sofa_install_libraries() instead.")
    sofa_install_libraries(PATHS ${library})
endfunction()


function(sofa_copy_libraries)
    set(multiValueArgs TARGETS LIBRARIES PATHS)
    cmake_parse_arguments("sofa_copy_libraries" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(targets ${sofa_copy_libraries_TARGETS})
    set(lib_paths ${sofa_copy_libraries_PATHS})
    if(sofa_copy_libraries_LIBRARIES)
        message(WARNING "sofa_copy_libraries: LIBRARIES parameter is deprecated, use PATHS instead.")
        list(APPEND lib_paths "${sofa_copy_libraries_LIBRARIES}")
    endif()

    foreach(target ${targets})
        if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
            foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
                get_target_property(target_location ${target} LOCATION_${CONFIG})
                list(APPEND lib_paths "${target_location}")
            endforeach()
        else() # Single-config generator (nmake)
            get_target_property(target_location ${target} LOCATION_${CMAKE_BUILD_TYPE})
            list(APPEND lib_paths "${target_location}")
        endif()
    endforeach()

    foreach(lib_path ${lib_paths})
        if(EXISTS ${lib_path})
            get_filename_component(LIB_NAME ${lib_path} NAME_WE)
            get_filename_component(LIB_PATH ${lib_path} PATH)

            file(GLOB SHARED_LIB
                "${LIB_PATH}/${LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
                "${LIB_PATH}/${LIB_NAME}[0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}"
                "${LIB_PATH}/${LIB_NAME}[0-9][0-9]${CMAKE_SHARED_LIBRARY_SUFFIX}")

            set(runtime_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
            if(NOT runtime_output_dir)
                set(runtime_output_dir ${CMAKE_BINARY_DIR}) # fallback
            endif()
            if(NOT EXISTS runtime_output_dir)
                # make sure runtime_output_dir exists before calling configure_file COPYONLY
                # otherwise it will not be treated as a directory
                file(MAKE_DIRECTORY "${runtime_output_dir}/")
            endif()

            if(EXISTS ${SHARED_LIB})
                if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (Visual Studio)
                    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
                        file(MAKE_DIRECTORY "${runtime_output_dir}/${CONFIG}/")
                        configure_file(${SHARED_LIB} "${runtime_output_dir}/${CONFIG}/" COPYONLY)
                    endforeach()
                else()                        # Single-config generator (nmake, ninja)
                    configure_file(${SHARED_LIB} "${runtime_output_dir}/" COPYONLY)
                endif()
            endif()
        endif()
    endforeach()
endfunction()


## to store which sources have been used for installed binaries
## these should be internal files and not delivered, but this is definitively useful
## when storing backups / demos across several repositories (e.g. sofa + plugins)
macro( sofa_install_git_version name sourcedir )
INSTALL( CODE
"
    find_package(Git REQUIRED)

    # the current commit hash should be enough
    # except if the git history changes...
    # so adding more stuff to be sure

    # get the current working branch
    execute_process(
      COMMAND \${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${sourcedir}
      OUTPUT_VARIABLE SOFA_GIT_BRANCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # get the current commit info (hash, author, date, comment)
    execute_process(
      COMMAND \${GIT_EXECUTABLE} log --format=medium -n 1
      WORKING_DIRECTORY ${sourcedir}
      OUTPUT_VARIABLE SOFA_GIT_INFO
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string( TOLOWER \"${name}\" name_lower )
    if( name_lower STREQUAL \"sofa\" )
        file(WRITE  \"${CMAKE_INSTALL_PREFIX}/git.version\" \"######## ${name} ########\nBranch: \${SOFA_GIT_BRANCH}\n\${SOFA_GIT_INFO}\n############\n\n\" )
    else()
        file(APPEND \"${CMAKE_INSTALL_PREFIX}/git.version\" \"######## ${name} ########\nBranch: \${SOFA_GIT_BRANCH}\n\${SOFA_GIT_INFO}\n############\n\n\" )
    endif()
"
)
endmacro()




#######################################################
################## DEPRECATED MACROS ##################
#######################################################


# sofa_install_targets
#
# package_name: Name of the package. One package can contain multiple targets. All the targets will be exported in ${package_name}Targets.
# the_targets: The targets to add to this package
# include_install_dir: Name of the INSTALLED directory that will contain headers
# (ARGV3) include_source_dir: Directory from which include tree will start (default: ${CMAKE_CURRENT_SOURCE_DIR})
# (ARGV4) example_install_dir: Name of the INSTALLED directory that will contain examples (default: share/sofa/${package_name}/examples)
macro(sofa_install_targets package_name the_targets include_install_dir)
    message(WARNING "Deprecated macro. Use 'sofa_add_targets_to_package' instead.")
    sofa_add_targets_to_package(
        PACKAGE_NAME ${package_name}
        TARGETS ${the_targets} AUTO_SET_TARGET_PROPERTIES
        INCLUDE_SOURCE_DIR "${ARGV3}"
        INCLUDE_INSTALL_DIR "${include_install_dir}"
        EXAMPLE_INSTALL_DIR "${ARGV4}"
        RELOCATABLE "${ARGV5}"
        )
endmacro()


# sofa_set_install_relocatable
#   TARGET MUST EXIST, TO BE CALLED AFTER add_library
# Content:
#   If building out of SOFA: does nothing.
#   If building through SOFA: call add_custom_target with custom commands to obtain a self-contained relocatable install.
#   Self-contained plugins are useful to build modular binaries: they do not "pollute" SOFA install
#   with self-contained plugins SOFA install will always look the same, no matter how many plugins are included.
# Effect:
#   add_custom_target will add the line 'set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${install_dir}/${name}")' at the top of the
#   plugin's cmake_install.cmake to force the plugin to be installed in it's own directory instead of in SOFA's install directory
#   (look at the build directory of any plugin to find an example of cmake_install.cmake).
function(sofa_set_install_relocatable target install_dir)
    message(WARNING "Deprecated function. Use 'sofa_set_target_install_relocatable' instead.")
    sofa_set_target_install_relocatable(${target} ${install_dir})
endfunction()


# sofa_write_package_config_files(Foo <version> <build-include-dirs>)
#
# Create CMake package configuration files
# - In the build tree:
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfig.cmake
#   - ${CMAKE_BINARY_DIR}/cmake/FooConfigVersion.cmake
# - In the install tree:
#   - lib/cmake/Foo/FooConfigVersion.cmake
#   - lib/cmake/Foo/FooConfig.cmake
#   - lib/cmake/Foo/FooTargets.cmake
#
# This macro factorizes boilerplate CMake code for the different
# packages in Sofa.  It assumes that there is a FooConfig.cmake.in
# file template in the same directory.  For example, if a package Foo
# depends on Bar and Baz, and creates the targets Foo and Qux, here is
# a typical FooConfig.cmake.in:
#
# @PACKAGE_INIT@
#
# find_package(Bar REQUIRED)
# find_package(Baz REQUIRED)
#
# if(NOT TARGET Qux)
# 	include("${CMAKE_CURRENT_LIST_DIR}/FooTargets.cmake")
# endif()
#
# check_required_components(Foo Qux)
macro(sofa_write_package_config_files package_name version)
    message(WARNING "Deprecated macro. Use 'sofa_create_package' instead.")
    sofa_create_package(
        PACKAGE_NAME ${package_name}
        PACKAGE_VERSION ${version}
        )
endmacro()


# - Create a target for SOFA plugin or module
# - write the package Config, Version & Target files
# - Deploy the headers, resources, scenes & examples
# - Replaces the now deprecated sofa_create_package macro
#
# sofa_generate_package(NAME VERSION TARGETS INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR)
#  NAME                - (input) the name of the generated package (usually ${PROJECT_NAME}).
#  VERSION             - (input) the package version (usually ${PROJECT_VERSION}).
#  TARGETS             - (input) list of targets to install. For standard plugins & modules, ${PROJECT_NAME}
#  INCLUDE_INSTALL_DIR - (input) [OPTIONAL] include directory (for Multi-dir install of header files).
#  INCLUDE_SOURCE_DIR  - (input) [OPTIONAL] install headers with same tree structure as source starting from this dir (defaults to ${CMAKE_CURRENT_SOURCE_DIR})
#
# Example:
# project(ExamplePlugin VERSION 1.0)
# find_package(SofaFramework)
# set(SOURCES_FILES  initExamplePlugin.cpp myComponent.cpp )
# set(HEADER_FILES   initExamplePlugin.h myComponent.h )
# add_library( ${PROJECT_NAME} SHARED ${SOURCE_FILES})
# target_link_libraries(${PROJECT_NAME} SofaCore)
# sofa_generate_package(NAME ${PROJECT_NAME} VERSION ${PROJECT_VERSION} TARGETS ${PROJECT_NAME} INCLUDE_INSTALL_DIR "sofa/custom/install/dir" INCLUDE_SOURCE_DIR src )
#
function(sofa_generate_package)
    set(oneValueArgs NAME VERSION INCLUDE_ROOT_DIR INCLUDE_INSTALL_DIR INCLUDE_SOURCE_DIR EXAMPLE_INSTALL_DIR RELOCATABLE)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments("ARG" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(WARNING "Deprecated macro. Use 'sofa_create_package_with_targets' instead.")
    sofa_create_package_with_targets(
        PACKAGE_NAME ${ARG_NAME}
        PACKAGE_VERSION ${ARG_VERSION}
        TARGETS ${ARG_TARGETS} AUTO_SET_TARGET_PROPERTIES
        INCLUDE_SOURCE_DIR "${ARG_INCLUDE_SOURCE_DIR}"
        INCLUDE_INSTALL_DIR "${ARG_INCLUDE_INSTALL_DIR}"
        EXAMPLE_INSTALL_DIR "${ARG_EXAMPLE_INSTALL_DIR}"
        RELOCATABLE "${ARG_RELOCATABLE}"
        )
endfunction()
