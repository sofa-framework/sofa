cmake_minimum_required(VERSION 2.8)

# group files (headers, sources, etc.)
get_cmake_property(properties VARIABLES)
foreach(property ${properties})
	if(NOT property STREQUAL "")
		string(REGEX MATCH ".+_FILES$" fileGroup ${property})
		if(fileGroup)
			GroupFiles("${fileGroup}")
		endif()
	endif()
endforeach()

# compile definitions
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};${COMPILE_DEFINES}")

# compile flags
if(WIN32)
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-wd4250 -wd4251 -wd4275 -wd4675 -wd4996 /bigobj")
endif()

# link dependencies
target_link_libraries(${PROJECT_NAME} ${LINK_DEPENDENCIES})
