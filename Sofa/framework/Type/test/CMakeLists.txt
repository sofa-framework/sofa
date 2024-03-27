cmake_minimum_required(VERSION 3.22)

project(Sofa.Type_test)

set(SOURCE_FILES
    Color_test.cpp
    Material_test.cpp
    Quater_test.cpp
    SVector_test.cpp
    vector_test.cpp
    MatTypes_test.cpp
    VecTypes_test.cpp
    fixed_array_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Type)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
