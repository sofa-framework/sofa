find_path(DIFFLIB_INCLUDE_DIR difflib.h
		  HINTS ${DIFFLIB_ROOT}
		  # If cross-compiling and typically use CMAKE_FIND_ROOT_PATH variable,
		  # each of its directory entry will be prepended to PATHS locations, and
		  # DIFFLIB_ROOT is set as an absolute path. So we have to disable this behavior
		  # for such external libs
		  NO_CMAKE_FIND_ROOT_PATH
	)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(DIFFLIB
		REQUIRED_VARS DIFFLIB_INCLUDE_DIR
)
mark_as_advanced(DIFFLIB_INCLUDE_DIR)
