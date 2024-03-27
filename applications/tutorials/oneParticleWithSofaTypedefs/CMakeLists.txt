cmake_minimum_required(VERSION 3.22)
project(oneParticleWithSofaTypedefs)

find_package(Sofa.GUI REQUIRED)

add_executable(${PROJECT_NAME} oneParticleWithSofaTypedefs.cpp)
target_link_libraries(${PROJECT_NAME} Sofa.GUI)
