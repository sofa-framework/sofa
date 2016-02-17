cmake_minimum_required(VERSION 2.8.12)

project(SofaPython_test)

set(SOURCE_FILES
    PyScene.cpp
    dummy.cpp
    python_test_list.cpp
)

if(SOFA_USE_C++11)
    list(APPEND SOURCE_FILES PythonScriptControllerHelper_test.cpp)
endif()

set(PYTHON_FILES
    python/dataVecResize.py
    python/testController.py
    python/test_PySceneWithPyCtlr.py
    python/test_Quaternion.py
    python/test_Tools.py
    python/test_mass.py
    python/test_units.py
    python/test_MeshLoader.py
)

if(SOFA_USE_C++11)
    list(APPEND PYTHON_FILES python/test_PythonScriptControllerHelper.py)
endif()

find_package(SofaTest)
find_package(SofaPython)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaPython SofaTest SofaGTestMain)
add_definitions("-DSOFAPYTHON_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
