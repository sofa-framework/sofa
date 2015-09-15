cmake_minimum_required(VERSION 2.8.12)
project(compositeObject)

find_package(SofaGui)

add_executable(${PROJECT_NAME} compositeObject.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)
