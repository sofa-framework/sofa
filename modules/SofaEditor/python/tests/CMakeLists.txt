cmake_minimum_required(VERSION 3.1)

project(PythonSofaEditor_test)

####################################################################################################
### Module dependencies
####################################################################################################
find_package(SofaPython 2.7 QUIET)
if(!SofaPython_FOUND)
    message(FATAL_ERROR "-- 'PythonSofaEditor_test' is disabled because 'SofaPython' is missing or not activated.")
    return()
endif()

find_package(SofaTest QUIET)
if(!SofaTest_FOUND)
    message(FATAL_ERROR "-- 'PythonSofaEditor_test' is disabled because 'SofaTest' is missing or not activated.")
    return()
endif()


####################################################################################################
### Building
####################################################################################################
set(SOURCE_FILES
    all_tests.cpp
)

set(PYTHON_FILES
    pyfiles/SofaEditor_test.py
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)
target_compile_definitions(${PROJECT_NAME} PRIVATE "PYTHON_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pyfiles/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

