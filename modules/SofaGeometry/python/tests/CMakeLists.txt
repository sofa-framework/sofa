cmake_minimum_required(VERSION 3.1)
project(PythonSofaGeometry_test)

####################################################################################################
### Module dependencies
####################################################################################################
find_package(SofaPython QUIET)
if(NOT SofaPython_FOUND)
    message("-- PythonSofaGeometry_test is disabled because 'SofaPython' is missing or not activated.")
    return()
endif()
find_package(SofaTest QUIET)
if(NOT SofaTest_FOUND)
    message("-- PythonSofaGeometry_test is disabled because 'SofaTest' is missing or not activated.")
    return()
endif()

message("-- PythonSofaEditor_test is enabled.")


####################################################################################################
### Building
####################################################################################################
set(SOURCE_FILES
    AllPythonSofaGeometry_test.cpp
)

set(PYTHON_FILES
    pyfiles/Plane_test.py
    pyfiles/Ray_test.py
    pyfiles/vector_test.py
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)
target_compile_definitions(${PROJECT_NAME} PRIVATE "PYTHON_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pyfiles/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

