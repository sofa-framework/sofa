cmake_minimum_required(VERSION 3.22)
project(myCfExport)

find_package(SofaSimulation)
find_package(SofaGeneral)

add_executable(${PROJECT_NAME} myCfExport.cpp)
target_link_libraries(${PROJECT_NAME} SofaGeneral SofaSimulationGraph)
# if(UNIX)
#     target_link_libraries(${PROJECT_NAME} dl)
# endif()
