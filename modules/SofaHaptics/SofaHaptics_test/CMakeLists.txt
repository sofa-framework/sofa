cmake_minimum_required(VERSION 3.1)

project(SofaHaptics_test)

sofa_find_package(SofaBaseUtils REQUIRED)
sofa_find_package(SofaLoader REQUIRED)
sofa_find_package(SofaGeneralLoader REQUIRED)
sofa_find_package(SofaHaptics REQUIRED)

set(SOURCE_FILES
    LCPForceFeedback_test.cpp)

add_definitions("-DSOFAHAPTICS_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaHaptics SofaBaseUtils)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
