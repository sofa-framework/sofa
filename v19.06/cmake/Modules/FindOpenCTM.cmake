#
# Try to find openCTM library and include path.
# Once done this will define
#
# OPENCTM_FOUND
# OPENCTM_INCLUDE_PATH
# OPENCTM_LIBRARY
# 
# lib directory of OpenCTM library should be linked to on of the path below under teh directory openctm

#include( Common )

IF (WIN32)
	FIND_PATH( OPENCTM_INCLUDE_PATH openctm/openctm.h
		DOC "The directory where openctm.h resides")
	FIND_LIBRARY( OPENCTM_LIBRARY_RELEASE
		NAMES openctm
		PATH_SUFFIXES Release
		DOC "The openctm library")
	FIND_LIBRARY( OPENCTM_LIBRARY_DEBUG
		NAMES openctmd
		PATH_SUFFIXES Debug
		DOC "The openctm library")
		
	IF(OPENCTM_LIBRARY_DEBUG)
		SET(OPENCTM_LIBRARY debug ${OPENCTM_LIBRARY_DEBUG} optimized ${OPENCTM_LIBRARY_RELEASE})
	ELSE()
		SET(OPENCTM_LIBRARY ${OPENCTM_LIBRARY_RELEASE})
	ENDIF()

ELSE (WIN32)
	FIND_PATH( OPENCTM_INCLUDE_PATH openctm/openctm.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where openctm.h resides")
	FIND_LIBRARY( OPENCTM_LIBRARY
		NAMES openctm/libopenctm.so
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The openctm library")
ENDIF (WIN32)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCTM DEFAULT_MSG OPENCTM_LIBRARY OPENCTM_INCLUDE_PATH)

mark_as_advanced(
  OPENCTM_INCLUDE_PATH
  OPENCTM_LIBRARY
)

