project(licenseUpdater)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)

if(NOT WIN32)
	add_definitions("-std=c++11")
endif()

find_package(Boost COMPONENTS system filesystem REQUIRED)

add_executable(${PROJECT_NAME} licenseUpdater.cpp)

target_link_libraries(${PROJECT_NAME}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
)
