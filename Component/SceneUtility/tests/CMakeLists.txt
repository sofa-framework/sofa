cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.SceneUtility_test)

set(SOURCE_FILES
    AddResourceRepository_test.cpp
    MakeAliasComponent_test.cpp
    MakeDataAliasComponent_test.cpp
    MessageHandlerComponent_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.SceneUtility)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
