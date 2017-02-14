cmake_minimum_required(VERSION 3.1)

project(SofaBaseCollision_test)

set(SOURCE_FILES
    BroadPhase_test.cpp
    OBB_test.cpp
    Sphere_test.cpp
    DefaultPipeline_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
