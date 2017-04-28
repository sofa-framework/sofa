cmake_minimum_required(VERSION 3.1)

project(SofaOpenglVisual_test)

set(SOURCE_FILES
    LightManager_test.cpp
    Light_test.cpp
    ClipPlane_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
