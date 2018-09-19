cmake_minimum_required(VERSION 3.1)
project(PythonModule_Sofa_test)

####################################################################################################
### Module dependencies
####################################################################################################
find_package(SofaPython3 QUIET)
if(NOT SofaPython3_FOUND)
    message("-- PythonModule_Sofa_test is disabled because 'SofaPython3' is missing or not activated.")
    return()
endif()

find_package(SofaTest QUIET)
if(NOT SofaTest_FOUND)
    message("-- PythonModule_Sofa_test is disabled because 'SofaTest' is missing or not activated.")
    return()
endif()

message("-- PythonModule_Sofa_test is enabled.")


####################################################################################################
### Building
####################################################################################################
set(SOURCE_FILES
    PythonModule_Sofa_test.cpp
)

set(PYTHON_FILES
    pyfiles/BaseData.py
    pyfiles/Base.py
    pyfiles/BaseObject.py
    pyfiles/Node.py
    pyfiles/ScriptController.py
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaPython3)
target_compile_definitions(${PROJECT_NAME} PRIVATE "PYTHON_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pyfiles/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

