cmake_minimum_required(VERSION 3.24)

include(FetchContent)

Set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
  FIND_PACKAGE_ARGS NAMES nlohmann_json
  )

FetchContent_MakeAvailable(nlohmann_json)