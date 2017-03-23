cmake_minimum_required(VERSION 3.1)

project(SofaRigid_test)

set(SOURCE_FILES
    RigidMapping_test.cpp
    RigidRigidMapping_test.cpp
    QuaternionIntegration_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
