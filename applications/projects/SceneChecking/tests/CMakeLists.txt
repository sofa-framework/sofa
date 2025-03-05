cmake_minimum_required(VERSION 3.22)

project(SceneChecking_test)

set(SOURCE_FILES SceneChecker_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SceneChecking)
