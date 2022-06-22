cmake_minimum_required(VERSION 2.8)
project(GKlib C)

option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" OFF)

get_filename_component(abs "." ABSOLUTE)
set(GKLIB_PATH ${abs})
unset(abs)
include(GKlibSystem.cmake)

include_directories(".")
if(MSVC)
  include_directories("win32")
  file(GLOB win32_sources RELATIVE "win32" "*.c")
else(MSVC)
  set(win32_sources, "")
endif(MSVC)

add_library(GKlib ${GKlib_sources} ${win32_sources})

if(UNIX)
  target_link_libraries(GKlib m)
endif(UNIX)

install(TARGETS GKlib
  ARCHIVE DESTINATION lib/${LINSTALL_PATH}
  LIBRARY DESTINATION lib/${LINSTALL_PATH})
install(FILES ${GKlib_includes} DESTINATION include/${HINSTALL_PATH})
