# Find the CHOLMOD library from SuiteSparse
#
# Behavior: first tries to find SuiteSparse CMake config files (SuiteSparse >= 7),
# then falls back to manual header/library search.
#
# Defines:
#   CHOLMOD_FOUND : True if CHOLMOD is found
#   CHOLMOD_INCLUDE_DIRS : CHOLMOD include directories
#   CHOLMOD_LIBRARIES : CHOLMOD libraries to link against
#
# Provides target SuiteSparse::CHOLMOD when using SuiteSparse config files,
# or a manual CHOLMOD::CHOLMOD imported target otherwise.

# Try SuiteSparse >= 7 CMake config first
find_package(SuiteSparse CONFIG QUIET COMPONENTS CHOLMOD)
if(TARGET SuiteSparse::CHOLMOD)
    set(CHOLMOD_FOUND TRUE)
    get_target_property(CHOLMOD_INCLUDE_DIRS SuiteSparse::CHOLMOD INTERFACE_INCLUDE_DIRECTORIES)
    set(CHOLMOD_LIBRARIES SuiteSparse::CHOLMOD)
    return()
endif()

# Try CHOLMOD standalone config
find_package(CHOLMOD CONFIG QUIET)
if(TARGET CHOLMOD::CHOLMOD)
    set(CHOLMOD_FOUND TRUE)
    get_target_property(CHOLMOD_INCLUDE_DIRS CHOLMOD::CHOLMOD INTERFACE_INCLUDE_DIRECTORIES)
    set(CHOLMOD_LIBRARIES CHOLMOD::CHOLMOD)
    return()
endif()

# Fallback: manual search
find_path(CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    PATH_SUFFIXES suitesparse SuiteSparse
)

find_library(CHOLMOD_LIBRARY
    NAMES cholmod
)

# CHOLMOD also needs SuiteSparse_config and AMD at minimum
find_library(SUITESPARSE_CONFIG_LIBRARY
    NAMES suitesparseconfig
)

find_library(AMD_LIBRARY
    NAMES amd
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD
    REQUIRED_VARS CHOLMOD_LIBRARY CHOLMOD_INCLUDE_DIR
)

if(CHOLMOD_FOUND)
    set(CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIR})
    set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY})
    if(SUITESPARSE_CONFIG_LIBRARY)
        list(APPEND CHOLMOD_LIBRARIES ${SUITESPARSE_CONFIG_LIBRARY})
    endif()
    if(AMD_LIBRARY)
        list(APPEND CHOLMOD_LIBRARIES ${AMD_LIBRARY})
    endif()

    if(NOT TARGET CHOLMOD::CHOLMOD)
        add_library(CHOLMOD::CHOLMOD UNKNOWN IMPORTED)
        set_target_properties(CHOLMOD::CHOLMOD PROPERTIES
            IMPORTED_LOCATION "${CHOLMOD_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CHOLMOD_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(CHOLMOD_INCLUDE_DIR CHOLMOD_LIBRARY SUITESPARSE_CONFIG_LIBRARY AMD_LIBRARY)
