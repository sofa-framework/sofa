cmake_minimum_required(VERSION 3.12)
project(oneTetrahedron)

sofa_find_package(SofaGui REQUIRED)
sofa_find_package(SofaImplicitOdeSolver REQUIRED)
sofa_find_package(SofaOpenglVisual REQUIRED)
sofa_find_package(SofaSimulationGraph REQUIRED)

add_executable(${PROJECT_NAME} oneTetrahedron.cpp)
target_link_libraries(${PROJECT_NAME} SofaGui SofaOpenglVisual SofaImplicitOdeSolver SofaSimulationGraph)
