cmake_minimum_required(VERSION 3.1)
project(sofaInfo)

find_package(SofaGeneral)
find_package(SofaAdvanced)
find_package(SofaMisc)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

add_executable(${PROJECT_NAME} sofaBatch.cpp)
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
if(UNIX)
    target_link_libraries(${PROJECT_NAME} dl)
endif()
