cmake_minimum_required(VERSION 3.1)
project(oneParticleWithSofaTypedefs)

find_package(SofaGui)

add_executable(${PROJECT_NAME} oneParticleWithSofaTypedefs.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)
