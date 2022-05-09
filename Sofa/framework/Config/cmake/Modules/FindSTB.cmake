find_path(STB_INCLUDE_DIR stb_image.h
		  HINTS ${STB_ROOT}
	)

include(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(STB
		REQUIRED_VARS STB_INCLUDE_DIR
)
mark_as_advanced(STB_INCLUDE_DIR)
