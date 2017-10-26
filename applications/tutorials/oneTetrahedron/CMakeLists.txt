cmake_minimum_required(VERSION 3.1)
project(oneTetrahedron)

find_package(SofaGui)

add_executable(${PROJECT_NAME} oneTetrahedron.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)
