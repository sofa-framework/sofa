cmake_minimum_required(VERSION 3.1)
project(findAlias)

find_package(SofaGeneral)
find_package(SofaMisc)
find_package(SofaBase)
find_package(SofaCommon)

add_executable(${PROJECT_NAME} findAlias.cpp)
target_link_libraries(${PROJECT_NAME} SofaGeneral SofaBase SofaCommon SofaMisc)
