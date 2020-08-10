# - Try to find ZMQ
# Once done this will define
# ZMQ_FOUND - System has ZMQ
# ZMQ_INCLUDE_DIRS - The ZMQ include directories
# ZMQ_LIBRARIES - The libraries needed to use ZMQ

find_path(ZMQ_INCLUDE_DIR zmq.h)
find_library(ZMQ_LIBRARY NAMES zmq)

# If lib not found, try searching in ZMQ_ROOT/lib (env and CMake vars)
if(NOT ZMQ_LIBRARY)
	file(GLOB libzmq LIST_DIRECTORIES false RELATIVE "$ENV{ZMQ_ROOT}/lib" "$ENV{ZMQ_ROOT}/lib/libzmq*")
    find_library(ZMQ_LIBRARY NAMES ${libzmq})
endif()
if(NOT ZMQ_LIBRARY)
	file(GLOB libzmq LIST_DIRECTORIES false RELATIVE "${ZMQ_ROOT}/lib" "${ZMQ_ROOT}/lib/libzmq*")
    find_library(ZMQ_LIBRARY NAMES ${libzmq})
endif()

set(ZMQ_LIBRARIES ${ZMQ_LIBRARY})
set(ZMQ_INCLUDE_DIRS ${ZMQ_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set ZMQ_FOUND to TRUE
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZMQ DEFAULT_MSG ZMQ_LIBRARY ZMQ_INCLUDE_DIR)
