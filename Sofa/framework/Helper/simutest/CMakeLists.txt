cmake_minimum_required(VERSION 3.12)

project(Sofa.Helper_simutest)

# Do not find dependencies explicitely, let CMake decide the appropriate time to do it
#sofa_find_package(SofaBase REQUIRED)

set(SOURCE_FILES
    AdvancedTimer_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Helper)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
