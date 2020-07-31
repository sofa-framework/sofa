# - Try to find Oscpack
# Once done this will define
# Oscpack_FOUND - System has Oscpack
# Oscpack_INCLUDE_DIRS - The Oscpack include directories
# Oscpack_LIBRARIES - The libraries needed to use Oscpack

find_path(Oscpack_INCLUDE_DIR osc/OscTypes.h)
find_library(Oscpack_LIBRARY Oscpack)

set(Oscpack_LIBRARIES ${Oscpack_LIBRARY})
set(Oscpack_INCLUDE_DIRS ${Oscpack_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set Oscpack_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Oscpack DEFAULT_MSG Oscpack_LIBRARY Oscpack_INCLUDE_DIR)
