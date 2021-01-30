cmake_minimum_required(VERSION 3.12)
project(oneParticle)

find_package(SofaSimulation REQUIRED)
find_package(SofaGui REQUIRED)

add_executable(${PROJECT_NAME} oneParticle.cpp)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph SofaGui)
