cmake_minimum_required(VERSION 3.12)

project(Sofa.DefaultType_simutest)

set(SOURCE_FILES
    TemplateAliases_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.DefaultType)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
