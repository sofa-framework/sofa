cmake_minimum_required(VERSION 3.22)

project(Sofa.GL_test)

set(SOURCE_FILES
    src/GLSLShader_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.GL)

add_test(NAME Sofa.GL_test COMMAND Sofa.GL_test)
