cmake_minimum_required(VERSION 3.12)
project(getDeprecatedComponents)

## Dependencies
find_package(SofaFramework REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} Sofa.Helper)
