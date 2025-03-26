cmake_minimum_required(VERSION 3.22)
project(SimpleAPIExamples)

sofa_find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Simulation REQUIRED)
sofa_find_package(Sofa.SimpleApi REQUIRED)
sofa_find_package(Sofa.GUI REQUIRED)

add_executable(fallingSOFA fallingSOFA.cpp)
target_link_libraries(fallingSOFA Sofa.SimpleApi Sofa.Simulation Sofa.Config Sofa.GUI)
