cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Engine.Transform_test)

set(SOURCE_FILES
    DifferenceEngine_test.cpp
    DilateEngine_test.cpp
    DisplacementMatrixEngine_test.cpp
    DisplacementTransformEngine_test.cpp
    Engine.Transform_DataUpdate_test.cpp
    IndexValueMapper_test.cpp
    ProjectiveTransformEngine_test.cpp
    SmoothMeshEngine_test.cpp
    TransformEngine_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Engine.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Engine.Transform)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
