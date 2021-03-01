cmake_minimum_required(VERSION 3.12)

project(SofaBaseMechanics_test)

find_package(SofaTest REQUIRED)

set(SOURCE_FILES
    UniformMass_test.cpp
    DiagonalMass_test.cpp
    MechanicalObject_test.cpp
    UniformMass_test.cpp
    BarycentricMapping_test.cpp
    )

if(SOFATEST_HAVE_SOFAPYTHON)
    add_definitions("-DSOFABASEMECHANICS_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
    list(APPEND SOURCE_FILES python_test_list.cpp)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
