cmake_minimum_required(VERSION 3.1)

project(SofaPython_test)

set(SOURCE_FILES
    PyScene.cpp
    dummy.cpp
    python_test_list.cpp
    memory.cpp
    PythonFactory_test.cpp
    SofaModule_test.cpp
    PythonBinding_test_list.cpp
    PythonScriptControllerHelper_test.cpp
    PythonScriptController_test.cpp
    SofaModule_test.cpp
)

set(PYTHON_FILES
    python/automaticNodeInitialization.py
    python/sysPathDuplicate.py
    python/dataVecResize.py
    python/testController.py
    python/moduleReload.py
    python/test_PySceneWithPyCtlr.py
    python/test_Quaternion.py
    python/test_Tools.py
    python/test_mass.py
    python/test_memory.py
    python/test_units.py
    python/test_SofaModule.py
    python/test_MeshLoader.py
    python/test_PythonFactory.py
    python/test_PythonScriptControllerHelper.py
    python/test_PythonScriptController.py
    python/test_BindingSofa.py
    python/test_BindingBase.py
    python/test_BindingData.py
    python/test_BindingLink.py
    python/unicodeData.py
)


find_package(SofaTest)
find_package(SofaPython)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaPython SofaTest SofaGTestMain)
add_definitions("-DSOFAPYTHON_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
