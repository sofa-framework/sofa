cmake_minimum_required(VERSION 3.12)
project(houseOfCards)

find_package(SofaGui REQUIRED)
find_package(SceneCreator REQUIRED)

add_executable(${PROJECT_NAME} houseOfCards.cpp)
target_link_libraries(${PROJECT_NAME} SofaGui SceneCreator)
