include_guard(GLOBAL)
include(CMakePackageConfigHelpers)
include(CMakeParseLibraryList)


function(debug_print_target_properties tgt)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

    # Convert command output into a CMake list
    STRING(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    STRING(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

    if(NOT TARGET ${tgt})
      message("There is no target named '${tgt}'")
      return()
    endif()

    foreach(prop ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})
        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(prop STREQUAL "LOCATION" OR prop MATCHES "^LOCATION_" OR prop MATCHES "_LOCATION$")
            continue()
        endif()
        # message ("Checking ${prop}")
        get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
        if (propval)
            get_target_property(propval ${tgt} ${prop})
            message ("${tgt} ${prop} = ${propval}")
        endif()
    endforeach(prop)
endfunction()

macro(__get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        __get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

function(sofa_get_all_targets var)
    set(targets)

    set(source_dir ${ARGV1}) #optional argument to define the source directory
    if(NOT DEFINED source_dir)
        set(source_dir "${CMAKE_CURRENT_SOURCE_DIR}") # Set a default value
    endif()

    __get_all_targets_recursive(targets ${source_dir})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()


macro(sofa_fetch_dependency name)
    #TODO: use this for plugins

    set(oneValueArgs GIT_TAG GIT_REPOSITORY FETCH_ENABLED )
    set(multiValueArgs "")
    set(options DONT_BUILD)
    cmake_parse_arguments("ARG" "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}")

    # Setup fetch directory
    set(fetched_dir "${CMAKE_BINARY_DIR}/external_directories/fetched/${name}" )
    set(build_directory "${CMAKE_BINARY_DIR}/external_directories/fetched/${name}-build")

    # Create option
    string(REPLACE "\." "_"  fixed_name ${name})
    string(TOUPPER ${fixed_name} upper_name)
    set(${upper_name}_GIT_REPOSITORY "${ARG_GIT_REPOSITORY}" CACHE STRING "Repository address" )
    set(${upper_name}_GIT_TAG "${ARG_GIT_TAG}" CACHE STRING "Branch or commit SHA to checkout" )
    set(${upper_name}_LOCAL_DIRECTORY "" CACHE STRING "Absolute path to a local folder containing the cloned repository")


    set(${name}_SOURCE_DIR "${fetched_dir}" CACHE STRING "" FORCE )

    if( "${${upper_name}_LOCAL_DIRECTORY}" STREQUAL "" AND NOT FETCHCONTENT_FULLY_DISCONNECTED AND NOT FETCHCONTENT_UPDATES_DISCONNECTED AND NOT "${ARG_FETCH_ENABLED}" STREQUAL "OFF")
        # Fetch
        message("Fetching dependency ${name} in ${fetched_dir}")
        message(STATUS "Checkout reference ${${upper_name}_GIT_TAG} from repository ${${upper_name}_GIT_REPOSITORY} ")

        #Generate temporary folder to store project that will fetch the sources
        if(NOT EXISTS ${fetched_dir}-temp)
            file(MAKE_DIRECTORY "${fetched_dir}-temp/")
        endif()


        file(WRITE ${fetched_dir}-temp/CMakeLists.txt "
        cmake_minimum_required(VERSION 3.22)
        include(ExternalProject)
        ExternalProject_Add(
            ${name}
            GIT_REPOSITORY ${${upper_name}_GIT_REPOSITORY}
            GIT_TAG ${${upper_name}_GIT_TAG}
            SOURCE_DIR ${fetched_dir}
            BINARY_DIR \"\"
            CONFIGURE_COMMAND \"\"
            BUILD_COMMAND \"\"
            INSTALL_COMMAND \"\"
            TEST_COMMAND \"\"
            GIT_CONFIG \"remote.origin.fetch=+refs/pull/*:refs/remotes/origin/pr/*\"
            )"
        )

        execute_process(COMMAND "${CMAKE_COMMAND}" -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY "${fetched_dir}-temp"
                RESULT_VARIABLE generate_exitcode
                OUTPUT_VARIABLE generate_logs ERROR_VARIABLE generate_logs)
        file(APPEND "${fetched_dir}-temp/logs.txt" "${generate_logs}")
        execute_process(COMMAND "${CMAKE_COMMAND}" --build .
                WORKING_DIRECTORY "${fetched_dir}-temp"
                RESULT_VARIABLE build_exitcode
                OUTPUT_VARIABLE build_logs ERROR_VARIABLE build_logs)
        file(APPEND "${fetched_dir}-temp/logs.txt" "${build_logs}")

        if(NOT generate_exitcode EQUAL 0 OR NOT build_exitcode EQUAL 0)
            message(SEND_ERROR "Failed to fetch external repository ${name}." "\nSee logs in ${fetched_dir}-temp/logs.txt")
        endif()
    elseif (NOT ${upper_name}_LOCAL_DIRECTORY STREQUAL "")
        if(EXISTS ${${upper_name}_LOCAL_DIRECTORY})
            message("${name}: Using local directory ${${upper_name}_LOCAL_DIRECTORY}.")
            set(fetched_dir "${${upper_name}_LOCAL_DIRECTORY}")
        else ()
            message(SEND_ERROR "${name}: Specified directory ${${upper_name}_LOCAL_DIRECTORY} doesn't exist." "\nPlease provide a directory containing the fetched project, or use option ${fetch_enabled} to automatically fetch it.")
        endif ()
    endif()


    # Add
    if(NOT ARG_DONT_BUILD AND  EXISTS "${fetched_dir}/.git" AND IS_DIRECTORY "${fetched_dir}/.git")
        set(${name}_BUILD_DIR "${build_directory}" CACHE STRING "" FORCE)
        add_subdirectory("${fetched_dir}" "${build_directory}")
    elseif(NOT ARG_DONT_BUILD AND NOT ${upper_name}_LOCAL_DIRECTORY STREQUAL "")
        message(SEND_ERROR "Directory ${${upper_name}_LOCAL_DIRECTORY} given in ${upper_name}_LOCAL_DIRECTORY doesn't seem to be a right github repository.")
    elseif (NOT ARG_DONT_BUILD AND FETCHCONTENT_FULLY_DISCONNECTED OR FETCHCONTENT_UPDATES_DISCONNECTED)
        message(SEND_ERROR "FETCHCONTENT_FULLY_DISCONNECTED or FETCHCONTENT_UPDATES_DISCONNECTED is ON but the dependency hasn't been fetched correctly before. Please reconnect fetching mechanism.")
    endif()
endmacro()

