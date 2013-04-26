cmake_minimum_required(VERSION 2.8)

if(TARGET ${PROJECT_NAME})
	# group files (headers, sources, etc.)
	get_directory_property(properties VARIABLES)
	foreach(property ${properties})
		if(NOT property STREQUAL "")
			string(REGEX MATCH "^.+_FILES$" fileGroup ${property})
			set(fileTopGroup)
			if(${fileGroup}_GROUP)
				set(fileTopGroup "${${fileGroup}_GROUP}")
			endif()
			if(fileGroup)
				GroupFiles("${fileGroup}" "${fileTopGroup}" "${GROUP_BASE_DIR}")
			endif()
		endif()
	endforeach()

	# include directories
	include_directories(${GLOBAL_INCLUDE_DIRECTORIES} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

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
	if(XBOX)
		set(COMPILER_FLAGS_OS_SPECIFIC "-wd4250 -wd4231 /GR /EHsc /bigobj")
	endif()
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_OS_SPECIFIC} ${COMPILER_FLAGS}")

	# link dependencies
	target_link_libraries(${PROJECT_NAME} ${ADDITIONAL_LINKER_DEPENDENCIES} ${LINKER_DEPENDENCIES})
	
	# store dependencies for further uses
	RegisterProjectDependencies(${PROJECT_NAME} ${ADDITIONAL_LINKER_DEPENDENCIES} ${LINKER_DEPENDENCIES})

	#link flags
	set(LINKER_FLAGS_OS_SPECIFIC "")
	set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${LINKER_FLAGS_OS_SPECIFIC} ${LINKER_FLAGS}")
	
	# output name suffixes
	set(LIBSUFFIX)
	if(WIN32)
		set(LIBSUFFIX "_${SOFA_VERSION_NUM}")
	endif()
	set(BUILDSUFFIX)
	get_target_property(BUILDTYPE ${PROJECT_NAME} TYPE)
	if(NOT "${BUILDTYPE}" STREQUAL "EXECUTABLE")
		set(BUILDSUFFIX ${LIBSUFFIX})
	endif()
	set_target_properties(${PROJECT_NAME} PROPERTIES
		DEBUG_POSTFIX "${BUILDSUFFIX}d"
		RELEASE_POSTFIX "${BUILDSUFFIX}"
		RELWITHDEBINFO_POSTFIX "${BUILDSUFFIX}"
		MINSIZEREL_POSTFIX "${BUILDSUFFIX}"
		COMPILE_DEFINITIONS_DEBUG "SOFA_LIBSUFFIX=${LIBSUFFIX}d"
		COMPILE_DEFINITIONS_RELEASE "SOFA_LIBSUFFIX=${LIBSUFFIX}"
		COMPILE_DEFINITIONS_RELWITHDEBINFO "SOFA_LIBSUFFIX=${LIBSUFFIX}"
		COMPILE_DEFINITIONS_MINSIZEREL "SOFA_LIBSUFFIX=${LIBSUFFIX}"
	)
endif()
