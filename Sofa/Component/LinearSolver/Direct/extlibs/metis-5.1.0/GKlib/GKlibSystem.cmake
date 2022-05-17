# Helper modules.
include(CheckFunctionExists)
include(CheckIncludeFile)

# Setup options.
option(METIS-GKLIB_GDB "enable use of GDB" OFF)
mark_as_advanced(METIS-GKLIB_GDB)
option(METIS-GKLIB_ASSERT "turn asserts on" OFF)
mark_as_advanced(METIS-GKLIB_ASSERT)
option(METIS-GKLIB_ASSERT2 "additional assertions" OFF)
mark_as_advanced(METIS-GKLIB_ASSERT2)
option(METIS-GKLIB_DEBUG "add debugging support" OFF)
mark_as_advanced(METIS-GKLIB_DEBUG)
option(METIS-GKLIB_GPROF "add gprof support" OFF)
mark_as_advanced(METIS-GKLIB_GPROF)
option(METIS-GKLIB_OPENMP "enable OpenMP support" OFF)
mark_as_advanced(METIS-GKLIB_OPENMP)
option(METIS-GKLIB_PCRE "enable PCRE support" OFF)
mark_as_advanced(METIS-GKLIB_PCRE)
option(METIS-GKLIB_GKREGEX "enable GKREGEX support" OFF)
mark_as_advanced(METIS-GKLIB_GKREGEX)
option(METIS-GKLIB_GKRAND "enable GKRAND support" OFF)
mark_as_advanced(METIS-GKLIB_GKRAND)

# Add compiler flags.
if(MSVC)
  set(GKlib_COPTS "/Ox")
  set(GKlib_COPTIONS "-DWIN32 -DMSC -D_CRT_SECURE_NO_DEPRECATE -DUSE_GKREGEX")
elseif(MINGW)
  set(GKlib_COPTS "-DUSE_GKREGEX")
else()
  set(GKlib_COPTS "-O3")
  set(GKlib_COPTIONS "-DLINUX -D_FILE_OFFSET_BITS=64")
endif(MSVC)
if(CYGWIN)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DCYGWIN")
endif(CYGWIN)
if(CMAKE_COMPILER_IS_GNUCC)
# GCC opts.
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -std=c99 -fno-strict-aliasing")
  if(NOT MINGW)
      set(GKlib_COPTIONS "${GKlib_COPTIONS} -fPIC")
  endif(NOT MINGW)
# GCC warnings.
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -Wall -pedantic -Wno-unused-but-set-variable -Wno-unused-variable -Wno-unknown-pragmas")
elseif(${CMAKE_C_COMPILER_ID} MATCHES "Sun")
# Sun insists on -xc99.
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -xc99")
endif(CMAKE_COMPILER_IS_GNUCC)

# Find OpenMP if it is requested.
if(METIS-GKLIB_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(GKlib_COPTIONS "${GKlib_COPTIONS} -D__OPENMP__ ${OpenMP_C_FLAGS}")
  else()
    message(WARNING "OpenMP was requested but support was not found")
  endif(OPENMP_FOUND)
endif(METIS-GKLIB_OPENMP)


# Add various definitions.
if(METIS-GKLIB_GDB)
  set(GKlib_COPTS "${GKlib_COPTS} -g")
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -Werror")
endif(METIS-GKLIB_GDB)


if(METIS-GKLIB_DEBUG)
  set(GKlib_COPTS "-g")
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DDEBUG")
endif(METIS-GKLIB_DEBUG)

if(METIS-GKLIB_GPROF)
  set(GKlib_COPTS "-pg")
endif(METIS-GKLIB_GPROF)

if(NOT METIS-GKLIB_ASSERT)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DNDEBUG")
endif(NOT METIS-GKLIB_ASSERT)

if(NOT METIS-GKLIB_ASSERT2)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DNDEBUG2")
endif(NOT METIS-GKLIB_ASSERT2)


# Add various options
if(METIS-GKLIB_PCRE)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -D__WITHPCRE__")
endif(METIS-GKLIB_PCRE)

if(METIS-GKLIB_GKREGEX)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DUSE_GKREGEX")
endif(METIS-GKLIB_GKREGEX)

if(METIS-GKLIB_GKRAND)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DUSE_GKRAND")
endif(METIS-GKLIB_GKRAND)


# Check for features.
check_include_file(execinfo.h HAVE_EXECINFO_H)
if(HAVE_EXECINFO_H)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DHAVE_EXECINFO_H")
endif(HAVE_EXECINFO_H)

check_function_exists(getline HAVE_GETLINE)
if(HAVE_GETLINE)
  set(GKlib_COPTIONS "${GKlib_COPTIONS} -DHAVE_GETLINE")
endif(HAVE_GETLINE)


# Custom check for TLS.
if(MSVC)
   set(GKlib_COPTIONS "${GKlib_COPTIONS} -D__thread=__declspec(thread)")
else()
  # This if checks if that value is cached or not.
  if("${HAVE_THREADLOCALSTORAGE}" MATCHES "^${HAVE_THREADLOCALSTORAGE}$")
    try_compile(HAVE_THREADLOCALSTORAGE
      ${CMAKE_BINARY_DIR}
      ${GKLIB_PATH}/conf/check_thread_storage.c)
    # if(HAVE_THREADLOCALSTORAGE)
    #   message(STATUS "checking for thread-local storage - found")
    # else()
    #   message(STATUS "checking for thread-local storage - not found")
    # endif()
  endif()
  if(NOT HAVE_THREADLOCALSTORAGE)
    set(GKlib_COPTIONS "${GKlib_COPTIONS} -D__thread=")
  endif()
endif()

# Finally set the official C flags.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GKlib_COPTIONS} ${GKlib_COPTS}")

# Find GKlib sources.
file(GLOB GKlib_sources ${GKLIB_PATH}/*.c)
file(GLOB GKlib_includes ${GKLIB_PATH}/*.h)
