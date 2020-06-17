cmake_minimum_required(VERSION 3.1)
project(sofaInitTimer)

find_package(SofaGeneral)
find_package(SofaMisc)

add_executable(${PROJECT_NAME} sofaInitTimer.cpp)
target_link_libraries(${PROJECT_NAME} SofaGeneral SofaMisc)
if(UNIX)
    target_link_libraries(${PROJECT_NAME} dl)
endif()
