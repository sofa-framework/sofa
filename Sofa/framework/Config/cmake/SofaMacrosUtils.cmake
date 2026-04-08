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

# guess if the git tag is a commit hash or an actual tag or a branch nane.
# heavily inspired by https://github.com/cpm-cmake/CPM.cmake/pull/130/changes#diff-6fcfee7f313f15253f88285a499e62cb58746d47ff2cfec173f1f4cd29feb44d
function(__is_git_tag_commit_hash GIT_TAG RESULT)
  string(LENGTH ${GIT_TAG} length)
  # full hash has 40 characters, and short hash has at least 7 characters.
  if (length LESS 7 OR length GREATER 40)
    SET(${RESULT} 0 PARENT_SCOPE)
  else()
    if (${GIT_TAG} MATCHES "^[a-fA-F0-9]+$")
      SET(${RESULT} 1 PARENT_SCOPE)
    else()
      SET(${RESULT} 0 PARENT_SCOPE)
    endif()
  endif()
endfunction()