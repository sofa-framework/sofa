# - Try to find VRPN
# Once done this will define
# VRPN_FOUND - System has VRPN
# VRPN_INCLUDE_DIRS - The VRPN include directories
# VRPN_LIBRARIES - The libraries needed to use VRPN

find_path ( VRPN_INCLUDE_DIR vrpn.h )
find_library ( VRPN_LIBRARY NAMES vrpn )

set ( VRPN_LIBRARIES ${VRPN_LIBRARY} )
set ( VRPN_INCLUDE_DIRS ${VRPN_INCLUDE_DIR} )

include ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set VRPN_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args ( VRPN DEFAULT_MSG VRPN_LIBRARY VRPN_INCLUDE_DIR )
