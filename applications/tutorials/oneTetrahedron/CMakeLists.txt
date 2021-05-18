cmake_minimum_required(VERSION 3.12)
project(oneTetrahedron)

sofa_find_package(SofaGui REQUIRED)
sofa_find_package(SofaComponentAll REQUIRED)

add_executable(${PROJECT_NAME} oneTetrahedron.cpp)
target_link_libraries(${PROJECT_NAME} SofaGui SofaComponentAll)
