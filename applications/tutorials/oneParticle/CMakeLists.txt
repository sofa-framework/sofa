cmake_minimum_required(VERSION 2.8.12)
project(oneParticle)

find_package(SofaSimulation)
find_package(SofaGui)

add_executable(${PROJECT_NAME} oneParticle.cpp)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph SofaGuiMain)
