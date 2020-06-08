cmake_minimum_required(VERSION 3.12)
project(mixedPendulum)

find_package(SofaGui)
find_package(SofaOpenglVisual REQUIRED)

add_executable(${PROJECT_NAME} mixedPendulum.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain SofaOpenglVisual)
