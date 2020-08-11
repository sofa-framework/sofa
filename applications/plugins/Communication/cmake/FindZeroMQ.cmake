# - Try to find ZeroMQ
# Once done this will define
# ZeroMQ_FOUND - System has ZeroMQ
# ZeroMQ_INCLUDE_DIRS - The ZeroMQ include directories
# ZeroMQ_LIBRARIES - The libraries needed to use ZeroMQ

find_package(ZeroMQ QUIET CONFIG
    PATHS "$ENV{ZMQ_ROOT}" "${ZMQ_ROOT}"
    )
    
if(NOT ZeroMQ_FOUND)
    find_path(ZeroMQ_INCLUDE_DIR zmq.h)
    find_library(ZeroMQ_LIBRARY NAMES zmq)

    # [Windows] If lib not found, try searching in ZeroMQ_ROOT/bin (env and CMake vars)
    if(WIN32 AND NOT ZeroMQ_LIBRARY)
        file(GLOB libzmq LIST_DIRECTORIES false RELATIVE "$ENV{ZeroMQ_ROOT}/lib" "$ENV{ZeroMQ_ROOT}/bin/libzmq*")
        find_library(ZeroMQ_LIBRARY NAMES ${libzmq})
    endif()
    if(WIN32 AND NOT ZeroMQ_LIBRARY)
        file(GLOB libzmq LIST_DIRECTORIES false RELATIVE "${ZeroMQ_ROOT}/lib" "${ZeroMQ_ROOT}/bin/libzmq*")
        find_library(ZeroMQ_LIBRARY NAMES ${libzmq})
    endif()
endif()

if(ZeroMQ_LIBRARY AND ZeroMQ_INCLUDE_DIR AND NOT TARGET libzmq)
    add_library(libzmq INTERFACE IMPORTED)
    set_target_properties(libzmq PROPERTIES
        INTERFACE_LINK_LIBRARIES "${ZeroMQ_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZeroMQ_INCLUDE_DIR}")
endif()

set(ZeroMQ_LIBRARIES ${ZeroMQ_LIBRARY})
set(ZeroMQ_INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set ZeroMQ_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZeroMQ DEFAULT_MSG ZeroMQ_LIBRARY ZeroMQ_INCLUDE_DIR)
