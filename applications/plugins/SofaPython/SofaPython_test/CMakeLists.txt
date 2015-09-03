cmake_minimum_required(VERSION 2.8)

project(SofaPython_test)

set(SOURCE_FILES
    PyScene.cpp
    dummy.cpp
    python_test_list.cpp
)

set(PYTHON_FILES
    python/dataVecResize.py
    python/testController.py
    python/test_PySceneWithPyCtlr.py
    python/test_Quaternion.py
    python/test_Tools.py
    python/test_mass.py
    python/test_units.py
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaPython SofaGTestMain)
add_definitions("-DSOFAPYTHON_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
