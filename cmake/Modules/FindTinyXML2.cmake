cmake_minimum_required(VERSION 3.24)

include(FetchContent)
FetchContent_Declare(
  tinyxml2
  URL https://github.com/leethomason/tinyxml2/archive/refs/tags/9.0.0.tar.gz
  URL_HASH MD5=afecd941107a8e74d3d1b4363cf52bd7
  FIND_PACKAGE_ARGS NAMES TinyXML2
  )
set(tinyxml2_SHARED_LIBS ON)
FetchContent_MakeAvailable(tinyxml2)
