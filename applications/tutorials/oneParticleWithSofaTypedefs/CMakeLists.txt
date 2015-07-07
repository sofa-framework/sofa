cmake_minimum_required(VERSION 2.8)
project(oneParticleWithSofaTypedefs)

find_package(SofaGui)

add_executable(${PROJECT_NAME} oneParticleWithSofaTypedefs.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaTutorials") # IDE folder
