cmake_minimum_required(VERSION 2.8)

if(TARGET ${PROJECT_NAME})
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

	# include directories
	include_directories(${GLOBAL_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

	get_directory_property(${PROJECT_NAME}_INCLUDE_DIR INCLUDE_DIRECTORIES)
	## put includes inside a CACHE variable for further uses
	set(${PROJECT_NAME}_INCLUDE_DIR ${${PROJECT_NAME}_INCLUDE_DIR} CACHE INTERNAL "${PROJECT_NAME} include path" FORCE)

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
	
	# store dependencies for further uses
	RegisterProjectDependencies(${PROJECT_NAME} ${ADDITIONAL_LINKER_DEPENDENCIES} ${LINKER_DEPENDENCIES})

	#link flags
	set(LINKER_FLAGS_OS_SPECIFIC "")
	set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${LINKER_FLAGS_OS_SPECIFIC} ${LINKER_FLAGS}")
endif()
