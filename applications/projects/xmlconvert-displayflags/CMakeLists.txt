cmake_minimum_required(VERSION 3.22)
project(xmlconvert-displayflags)

find_package(TinyXML2 REQUIRED)

add_executable(${PROJECT_NAME} xmlvisitor.h xmlvisitor.cpp main.cpp)
target_link_libraries(${PROJECT_NAME} tinyxml2::tinyxml2 Sofa.Core)
