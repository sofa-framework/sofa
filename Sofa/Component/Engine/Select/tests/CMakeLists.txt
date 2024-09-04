cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Engine.Select_test)

set(SOURCE_FILES
    BoxROI_test.cpp
    ComplementaryROI_test.cpp
    Engine.Select_DataUpdate_test.cpp
    IndicesFromValues_test.cpp
    MeshROI_test.cpp
    MeshSubsetEngine_test.cpp
    PlaneROI_test.cpp
    SphereROI_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Engine.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Engine.Select)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
