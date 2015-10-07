cmake_minimum_required(VERSION 2.8.12)
project(generateDoc)

find_package(SofaGeneral)
find_package(SofaAdvanced)
find_package(SofaMisc)

add_executable(${PROJECT_NAME} generateDoc.h generateDoc.cpp Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
