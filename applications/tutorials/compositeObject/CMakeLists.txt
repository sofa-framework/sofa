cmake_minimum_required(VERSION 3.12)
project(compositeObject)

find_package(SofaGui REQUIRED)
find_package(SofaMiscMapping REQUIRED)

add_executable(${PROJECT_NAME} compositeObject.cpp)
target_link_libraries(${PROJECT_NAME} SofaGui SofaMiscMapping)
