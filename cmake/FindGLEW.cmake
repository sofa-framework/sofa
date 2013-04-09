# This module defines the following variables:
#  GLEW_INCLUDE_DIRS - include directories for GLEW
#  GLEW_LIBRARIES - libraries to link against GLEW
#  GLEW_FOUND - true if GLEW has been found and can be used


if(WIN32)
	set(GLEW_LIBRARIES "glew32")
	set(GLEW_FOUND CACHE INTERNAL TRUE)
endif(WIN32)

if(UNIX)
        find_path(GLEW_INCLUDE_DIR GL/glew.h)
        find_library(GLEW_LIBRARIES NAMES glew64 GLEW glew glew32)

	if(EXISTS "${GLEW_INCLUDE_DIR}/GL/glew.h" AND EXISTS ${GLEW_LIBRARIES})
		if (NOT DEFINED GLEW_FOUND)
			message(STATUS "GLEW Lib found: ${GLEW_LIBRARIES} ${GLEW_INCLUDE_DIR}")
		endif(NOT DEFINED GLEW_FOUND)
		set(GLEW_FOUND CACHE INTERNAL TRUE)
		include_directories(${GLEW_INCLUDE_DIR})
	else()
		message(FATAL_ERROR "GLEW NOT FOUND ${GLEW_LIBRARIES} ${GLEW_INCLUDE_DIR}")
		set(GLUE_FOUND FALSE)
	endif()

endif(UNIX)




mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARIES)
