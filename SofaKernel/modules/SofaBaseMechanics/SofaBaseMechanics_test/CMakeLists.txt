cmake_minimum_required(VERSION 3.1)

project(SofaBaseMechanics_test)

set(SOURCE_FILES
    UniformMass_test.cpp
    DiagonalMass_test.cpp
    MechanicalObject_test.cpp
    UniformMass_test.cpp
    )



find_package(SofaPython QUIET)
if(SofaPython_FOUND)
    add_definitions("-DSOFABASEMECHANICS_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
    list(APPEND SOURCE_FILES python_test_list.cpp)
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)
if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)
endif()

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
