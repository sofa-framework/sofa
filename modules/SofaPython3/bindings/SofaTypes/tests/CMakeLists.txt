cmake_minimum_required(VERSION 3.1)
project(PythonModule_SofaTypes_test)

####################################################################################################
### Module dependencies
####################################################################################################
find_package(SofaPython3 QUIET)
if(NOT SofaPython3_FOUND)
    message("-- PythonModule_SofaTypes_test is disabled because 'SofaPython3' is missing or not activated.")
    return()
endif()

message("-- PythonModule_SofaTypes_test is enabled.")


####################################################################################################
### Building
####################################################################################################
set(SOURCE_FILES
    PythonModule_SofaTypes_test.cpp
)

set(PYTHON_FILES
#    pyfiles/quaternion_test.py
#    pyfiles/matrix_test.py
    pyfiles/vector_test.py
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaPython3 SofaGTestMain)
target_compile_definitions(${PROJECT_NAME} PRIVATE "PYTHON_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pyfiles/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

