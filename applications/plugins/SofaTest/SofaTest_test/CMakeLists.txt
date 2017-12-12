cmake_minimum_required(VERSION 3.1)
project(SofaTest_test)

set(HEADER_FILES
    Regression_test.h
)

set(SOURCE_FILES
    Regression_test.cpp
)

set(TEST_FILES
    scenes/damping.py
    python_tests/python_test_test.py
    )

find_package(SofaPython QUIET)

if(SofaPython_FOUND)
    add_definitions("-DSOFATEST_PYTHONTESTS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python_tests\"")
    add_definitions("-DSOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    target_link_libraries(SofaPython)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${TEST_FILES})
add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)
endif()
