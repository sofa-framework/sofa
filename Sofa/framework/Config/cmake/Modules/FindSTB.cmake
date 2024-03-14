find_path(STB_INCLUDE_DIR stb_image.h
		  HINTS ${STB_ROOT}
		  # If cross-compiling and typically use CMAKE_FIND_ROOT_PATH variable,
		  # each of its directory entry will be prepended to PATHS locations, and
		  # STB_ROOT is set as an absolute path. So we have to disable this behavior
		  # for such external libs
		  NO_CMAKE_FIND_ROOT_PATH
	)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(STB
		REQUIRED_VARS STB_INCLUDE_DIR
)
mark_as_advanced(STB_INCLUDE_DIR)
