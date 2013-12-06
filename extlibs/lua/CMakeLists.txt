include(${SOFA_CMAKE_DIR}/preProject.cmake)

project ( lua C )
cmake_minimum_required ( VERSION 2.8 )
include ( cmake/dist.cmake )

## CONFIGURATION
# Default configuration (we assume POSIX by default)
# set ( SOFA-EXTERNAL_LUA_PATH "SOFA-EXTERNAL_LUA_PATH" CACHE STRING "Environment variable to use as package.path." )
set ( LUA_CPATH "LUA_CPATH" CACHE STRING "Environment variable to use as package.cpath." )
set ( LUA_INIT "LUA_INIT" CACHE STRING "Environment variable for initial script." )
set ( LUA_IDSIZE 60 CACHE NUMBER "gives the maximum size for the description of the source." )
set ( LUA_DIRSEP "/" )
set ( LUA_MODULE_SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX} )
set ( LUA_LDIR ${INSTALL_LMOD} )
set ( LUA_CDIR ${INSTALL_CMOD} )
set ( LUA_DIR ${CMAKE_INSTALL_PREFIX} CACHE STRING "Destination from which modules will be resolved. See INSTALL_LMOD and INSTALL_CMOD." )
set ( SOFA-EXTERNAL_LUA_PATH_DEFAULT "./?.lua;${LUA_DIR}${LUA_LDIR}/?.lua;${LUA_DIR}${LUA_LDIR}/?/init.lua" )
set ( LUA_CPATH_DEFAULT "./?${LUA_MODULE_SUFFIX};${LUA_DIR}${LUA_CDIR}/?${LUA_MODULE_SUFFIX};${LUA_DIR}${LUA_CDIR}/loadall${LUA_MODULE_SUFFIX}" )
option ( LUA_ANSI "Use only ansi features." OFF )
option ( LUA_COMPAT_ALL "Enable backwards compatibility options." ON )

if ( WIN32 AND NOT CYGWIN )
  # Windows systems
  option ( LUA_WIN "Windows specific build." OFF )
  option ( LUA_BUILD_WLUA "Build wLua interpretter without console output." OFF )
  option ( LUA_BUILD_AS_DLL "Build Lua library as Dll." ${BUILD_SHARED_LIBS} )
  
  # Paths (Double escapes ne  option needed)
  set ( LUA_DIRSEP "\\\\" )
  string ( REPLACE " /" ${LUA_DIRSEP} LUA_DIR "${LUA_DIR}" )
  string ( REPLACE "/" ${LUA_DIRSEP} LUA_LDIR "${LUA_LDIR}" )
  string ( REPLACE "/" ${LUA_DIRSEP} LUA_CDIR "${LUA_CDIR}" )
  string ( REPLACE "/" ${LUA_DIRSEP} SOFA-EXTERNAL_LUA_PATH_DEFAULT "${SOFA-EXTERNAL_LUA_PATH_DEFAULT}" )
  string ( REPLACE "/" ${LUA_DIRSEP} LUA_CPATH_DEFAULT "${LUA_CPATH_DEFAULT}" )
else ( )
  # Posix systems (incl. Cygwin)
  option ( LUA_USE_POSIX "Use POSIX features." ON )
  option ( LUA_USE_DLOPEN "Use dynamic linker to load modules." ON )
  option ( LUA_USE_MKSTEMP "Use mkstep." ON )
  option ( LUA_USE_ISATTY "Use tty." ON )
  option ( LUA_USE_POPEN "Use popen." ON )
  option ( LUA_USE_ULONGJMP "Use ulongjmp" ON )
  option ( LUA_USE_GMTIME_R "Use GTIME_R" ON )
  # Apple and Linux specific
  if ( LINUX OR APPLE )
    option ( LUA_USE_STRTODHEX "Assume 'strtod' handles hexa formats" ON )
    option ( LUA_USE_AFORMAT "Assume 'printf' handles 'aA' specifiers" ON )
    option ( LUA_USE_LONGLONG "Assume support for long long" ON )
  endif ( )
endif ( )

## SETUP
# Optional libraries
find_package ( Readline )
if ( READLINE_FOUND )
  option ( LUA_USE_READLINE "Use readline in the Lua CLI." ON )
endif ( )

find_package ( Curses )
if ( CURSES_FOUND )
  option ( LUA_USE_CURSES "Use curses in the Lua CLI." ON )
endif ( )

# Setup needed variables and libraries
if ( LUA_USE_POSIX )
  # On POSIX Lua links to standard math library "m"
  list ( APPEND LIBS m )
endif ( )

if ( LUA_USE_DLOPEN )
  # Link to dynamic linker library "dl"
	find_library ( DL_LIBRARY NAMES dl )
	if ( DL_LIBRARY )
  	list ( APPEND LIBS ${DL_LIBRARY} )
	endif ( )
endif ( )

if ( LUA_USE_READLINE )
  # Add readline
  include_directories ( ${READLINE_INCLUDE_DIR} )
  list ( APPEND LIBS ${READLINE_LIBRARY} )
endif ( )

if ( LUA_USE_CURSES )
  # Add curses
  include_directories ( ${CURSES_INCLUDE_DIR} )
  list ( APPEND LIBS ${CURSES_LIBRARY} )
endif ( )

## SOURCES
# Generate luaconf.h
#configure_file ( src/luaconf.h.in ${CMAKE_CURRENT_BINARY_DIR}/luaconf.h )

# Sources and headers
include_directories ( src ${CMAKE_CURRENT_BINARY_DIR} )

set (SOURCE_FILES src/lapi.c  src/lbaselib.c src/ldblib.c   src/ldo.c   src/lfunc.c    src/linit.c    src/llex.c  src/lmem.c  src/lobject.c  src/loslib.c  src/lstate.c
src/lstrlib.c  src/ltablib.c  src/lua.c   src/lundump.c  src/lzio.c src/lauxlib.c  src/lcode.c   src/ldebug.c   src/ldump.c   src/lgc.c   src/liolib.c  src/lmathlib.c
src/loadlib.c  src/lopcodes.c src/lparser.c  src/lstring.c  src/ltable.c   src/ltm.c   src/lvm.c   src/print.c )

set (HEADER_FILES src/lapi.h  src/lcode.h  src/ldo.h   src/lgc.h  src/llimits.h  src/lobject.h  src/lparser.h  src/lstring.h  src/ltm.h   src/luaconf.h
src/lundump.h  src/lzio.h src/lauxlib.h  src/ldebug.h   src/lfunc.h    src/llex.h     src/lmem.h     src/lopcodes.h src/lstate.h   src/ltable.h   src/lua.h  src/lualib.h  src/lvm.h )


## BUILD
# Create lua library
add_library ( liblua ${SOURCE_FILES} ${LUA_DLL_RC} ${LUA_DEF} )
target_link_libraries ( liblua ${LIBS} )
set_target_properties ( liblua PROPERTIES OUTPUT_NAME lua CLEAN_DIRECT_OUTPUT 1 )
if ( LUA_BUILD_AS_DLL )
  set_target_properties ( liblua PROPERTIES COMPILE_DEFINITIONS LUA_BUILD_AS_DLL )
endif ()

#add_executable ( lua ${SRC_LUA} src/lua.rc )
add_executable ( lua src/lua.c )
target_link_libraries ( lua liblua )

#add_executable ( luac ${SRC_CORE} ${SRC_LIB} ${SRC_LUAC} src/luac.rc )
#add_executable ( luac ${SOURCE_FILES} ${SRC_LUAC} )
#target_link_libraries ( luac ${LIBS} )

# On windows a variant of the lua interpreter without console output needs to be built
if ( LUA_BUILD_WLUA )
  add_executable ( wlua WIN32 src/wmain.c ${SRC_LUA} src/lua.rc )
  target_link_libraries ( wlua liblua )
  install_executable ( wlua )
endif ( )

#install_executable ( lua luac )
#install_library ( liblua )
#install_data ( README.md )
#install_lua_module ( strict etc/strict.lua )
#install_header ( src/lua.h src/lualib.h src/lauxlib.h src/lua.hpp ${CMAKE_CURRENT_BINARY_DIR}/luaconf.h )
#install_header ( src/lua.h src/lualib.h src/lauxlib.h src/lua.hpp)
#install_doc ( doc/ )
#install_foo ( etc/ )
#install_test ( test/ )


include(${SOFA_CMAKE_DIR}/postProject.cmake)

