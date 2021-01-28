cmake_minimum_required(VERSION 3.12)
project(sofaProjectExample)

find_package(SofaGui REQUIRED)
find_package(SofaOpenglVisual REQUIRED)

add_executable(${PROJECT_NAME} Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaGui SofaOpenglVisual)
