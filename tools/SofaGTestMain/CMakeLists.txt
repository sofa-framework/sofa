cmake_minimum_required(VERSION 2.8.12)

find_package(SofaSimulation)

include_directories("../extlibs/gtest/include")
link_directories("../extlibs/gtest/lib")

if(WIN32)
    # MSVC2012 has some troubles with the way gtest use the STL, this preprocessor macro fix this issue
    add_definitions("-D_VARIADIC_MAX=10")
endif()

add_library(SofaGTestMain SofaGTestMain.cpp)
target_link_libraries(SofaGTestMain gtest SofaSimulationTree SofaSimulationGraph)
