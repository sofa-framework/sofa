cmake_minimum_required(VERSION 2.8)
project(sofaInitTimer)

find_package(SofaGeneral)
find_package(SofaAdvanced)
find_package(SofaMisc)

add_executable(${PROJECT_NAME} sofaInitTimer.cpp)
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
if(UNIX)
    target_link_libraries(${PROJECT_NAME} dl)
endif()
