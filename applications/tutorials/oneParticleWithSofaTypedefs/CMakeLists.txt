cmake_minimum_required(VERSION 3.12)
project(oneParticleWithSofaTypedefs)

find_package(SofaGui REQUIRED)

add_executable(${PROJECT_NAME} oneParticleWithSofaTypedefs.cpp)
target_link_libraries(${PROJECT_NAME} SofaGui)
