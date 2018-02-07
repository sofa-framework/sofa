# - Try to find Oscpack
# Once done this will define
# Oscpack_FOUND - System has Oscpack
# Oscpack_INCLUDE_DIRS - The Oscpack include directories
# Oscpack_LIBRARIES - The libraries needed to use Oscpack

find_path( Oscpack_INCLUDE_DIR osc/OscTypes.h )

find_library(Oscpack_LIBRARY Oscpack )

if(Oscpack_INCLUDE_DIR AND Oscpack_LIBRARY)
    set(Oscpack_FOUND 1)
    set(Oscpack_LIBRARIES ${Oscpack_LIBRARY})
    set(Oscpack_INCLUDE_DIRS ${Oscpack_INCLUDE_DIR})
else(Oscpack_INCLUDE_DIR AND Oscpack_LIBRARY)
    MESSAGE(FATAL_ERROR "CANNOT FIND Oscpack_INCLUDE_DIR AND Oscpack_LIBRARY" )
endif(Oscpack_INCLUDE_DIR AND Oscpack_LIBRARY)
