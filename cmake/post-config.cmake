cmake_minimum_required(VERSION 2.8)

# group files (headers, sources, etc.)
get_cmake_property(properties VARIABLES)
foreach(property ${properties})
	if(NOT property STREQUAL "")
		string(REGEX MATCH ".+_FILES$" fileGroup ${property})
		set(fileTopGroup "${${fileGroup}_GROUP}")
		if(fileGroup)
			GroupFiles("${fileGroup}" "${fileTopGroup}")
		endif()
	endif()
endforeach()

if(TARGET ${PROJECT_NAME})

	set(allDependenciesIncludeDirs)

        # put includes inside a CACHE variable
        set(linkerDependencies ${ADDITIONAL_LINKER_DEPENDENCIES} ${LINKER_DEPENDENCIES})
        # includes from this target
        unset(${PROJECT_NAME}_INCLUDE_PATH)
        get_directory_property(${PROJECT_NAME}_INCLUDE_PATH INCLUDE_DIRECTORIES)
        #message(STATUS "Private include path for ${PROJECT_NAME} : ${PROJECT_NAME}_INCLUDE_PATH = ${${PROJECT_NAME}_INCLUDE_PATH}")
        list(APPEND ${PROJECT_NAME}_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
        # includes from dependencies
        foreach(linkerDependency ${linkerDependencies})
                #message(STATUS "Checking variable ${linkerDependency}_INCLUDE_PATH = ${${linkerDependency}_INCLUDE_PATH}")
                if(TARGET ${linkerDependency})
                        #message(STATUS "Checking variable ${linkerDependency}_INCLUDE_PATH")
                        if(DEFINED ${linkerDependency}_INCLUDE_PATH)
                                #message(STATUS "dependency include found for ${PROJECT_NAME} : ${linkerDependency}_INCLUDE_PATH = ${${linkerDependency}_INCLUDE_PATH}")
                                list(APPEND ${PROJECT_NAME}_INCLUDE_PATH "${${linkerDependency}_INCLUDE_PATH}")
                                list(REMOVE_DUPLICATES ${PROJECT_NAME}_INCLUDE_PATH)
                        endif(DEFINED ${linkerDependency}_INCLUDE_PATH)
                endif(TARGET ${linkerDependency})
        endforeach()
        set(${PROJECT_NAME}_INCLUDE_PATH ${${PROJECT_NAME}_INCLUDE_PATH} CACHE INTERNAL "${PROJECT_NAME} include path" FORCE)

        #message(STATUS "Include path for ${PROJECT_NAME} : ${PROJECT_NAME}_INCLUDE_PATH = ${${PROJECT_NAME}_INCLUDE_PATH}")
        include_directories(${${PROJECT_NAME}_INCLUDE_PATH})

	# compile definitions
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_COMPILER_DEFINES};${ADDITIONAL_COMPILER_DEFINES};${COMPILER_DEFINES}")

	# compile flags
	set(COMPILER_FLAGS_OS_SPECIFIC "")
	if(WIN32)
		set(COMPILER_FLAGS_OS_SPECIFIC "-wd4250 -wd4251 -wd4275 -wd4675 -wd4996 /bigobj")
	endif()
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_OS_SPECIFIC} ${COMPILER_FLAGS}")

	# link dependencies
	target_link_libraries(${PROJECT_NAME} ${ADDITIONAL_LINKER_DEPENDENCIES} ${LINKER_DEPENDENCIES})

	#link flags
	set(LINKER_FLAGS_OS_SPECIFIC "")
	if(WIN32)
		set(LINKER_FLAGS_OS_SPECIFIC "")
	endif()
	set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${LINKER_FLAGS_OS_SPECIFIC} ${LINKER_FLAGS}")

endif()
