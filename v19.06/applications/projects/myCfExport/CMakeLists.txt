cmake_minimum_required(VERSION 3.1)
project(myCfExport)

find_package(SofaSimulation)
find_package(SofaGeneral)
find_package(SofaAdvanced)
find_package(SofaMisc)

add_executable(${PROJECT_NAME} myCfExport.cpp)
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc SofaSimulationGraph)
# if(UNIX)
#     target_link_libraries(${PROJECT_NAME} dl)
# endif()
