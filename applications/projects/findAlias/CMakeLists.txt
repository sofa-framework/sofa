cmake_minimum_required(VERSION 3.1)
project(findAlias)

find_package(SofaGeneral)
find_package(SofaAdvanced)
find_package(SofaMisc)
find_package(SofaComponentBase)
find_package(SofaComponentCommon)

add_executable(${PROJECT_NAME} findAlias.cpp)
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral SofaComponentBase SofaComponentCommon SofaComponentAdvanced SofaComponentMisc)
