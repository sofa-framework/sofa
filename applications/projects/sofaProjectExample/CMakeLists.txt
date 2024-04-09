cmake_minimum_required(VERSION 3.22)
project(sofaProjectExample)

find_package(SofaSimulationGraph REQUIRED)
sofa_find_package(SofaComponentAll REQUIRED)
sofa_find_package(SofaGLFW REQUIRED)

add_executable(${PROJECT_NAME} Main.cpp)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph SofaComponentAll SofaGLFW)
