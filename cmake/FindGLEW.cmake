# This module defines the following variables:
#  GLEW_INCLUDE_DIRS - include directories for GLEW
#  GLEW_LIBRARIES - libraries to link against GLEW
#  GLEW_FOUND - true if GLEW has been found and can be used

#=============================================================================
# Copyright 2012 Benjamin Eikel
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

if (NOT DEFINED GLEW_FOUND)

	if(WIN32)
		set(GLEW_LIBRARIES "glew32")
	endif(WIN32)

	if(UNIX)
		if(APPLE)
			# on Mac, GLEW is in the dependency package
			set(GLEW_LIBRARIES "${SOFA_LIB_OS_DIR}/libglew.dylib")
			set(GLEW_INCLUDE_DIR "${SOFA_INC_DIR}")
		else(APPLE)
			find_path(GLEW_INCLUDE_DIR GL/glew.h)
			find_library(GLEW_LIBRARIES NAMES glew64 GLEW glew glew32)
		endif(APPLE)
	endif(UNIX)



	if(EXISTS "${GLEW_INCLUDE_DIR}/GL/glew.h" AND EXISTS ${GLEW_LIBRARIES})
		message(STATUS "GLEW Lib found: ${GLEW_LIBRARIES} ${GLEW_INCLUDE_DIR}")
		set(GLEW_FOUND SHARED INTERNAL TRUE)
	else()
		message(FATAL_ERROR "GLEW NOT FOUND ${GLEW_LIBRARIES} ${GLEW_INCLUDE_DIR}")
		set(GLUE_FOUND FALSE)
	endif()



endif(NOT DEFINED GLEW_FOUND)

mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARIES)
