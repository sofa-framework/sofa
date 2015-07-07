cmake_minimum_required(VERSION 2.8)
project(houseOfCards)

find_package(SofaGui)
find_package(SceneCreator)

add_executable(${PROJECT_NAME} houseOfCards.cpp)
target_link_libraries(${PROJECT_NAME} SofaGuiMain SceneCreator)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaTutorials") # IDE folder
