cmake_minimum_required(VERSION 2.8)
project(glutOnePick)

find_package(SofaSimpleGUI)

add_executable(${PROJECT_NAME} glutOnePick.cpp oneTetra.h)
target_link_libraries(${PROJECT_NAME} SofaSimpleGUI)
