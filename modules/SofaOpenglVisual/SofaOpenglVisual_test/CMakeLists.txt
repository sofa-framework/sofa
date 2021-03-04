cmake_minimum_required(VERSION 3.12)

project(SofaOpenglVisual_test)

set(SOURCE_FILES
    OglLabel_test.cpp
    )

## MAC-OS crash when we are testing opengl feature.
if(NOT CMAKE_SYSTEM_NAME STREQUAL Darwin)
  list(APPEND SOURCE_FILES
     Light_test.cpp
     LightManager_test.cpp
     ClipPlane_test.cpp)
endif()

sofa_find_package(SofaBase REQUIRED)
sofa_find_package(SofaOpenglVisual REQUIRED)
sofa_find_package(SofaBaseMechanics REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaOpenglVisual SofaBaseMechanics)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
