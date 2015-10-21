cmake_minimum_required(VERSION 2.8)
project(sofaProjectExample)

find_package(SofaGui)

add_executable(${PROJECT_NAME} Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)
