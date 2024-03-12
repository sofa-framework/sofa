cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Engine.Generate_test)

set(SOURCE_FILES
    Engine.Generate_DataUpdate_test.cpp
    ExtrudeEdgesAndGenerateQuads_test.cpp
    JoinPoints_test.cpp
    MergePoints_test.cpp
    RandomPointDistributionInSurface_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Engine.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Engine.Generate)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
