include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    dummy.cpp
)

if(APPLE)
    # set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")

if(SOFA-PLUGIN_SOFAPYTHON)
    AddCompilerDefinitions("SOFAPYTHON_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    list(APPEND SOURCE_FILES "PyScene.cpp")
    AddLinkerDependencies(SofaPython)

    set(PYTHON_FILES
        python/dataVecResize.py
        python/testController.py
        python/test_PySceneWithPyCtlr.py
        python/test_Quaternion.py
        python/test_Tools.py
        python/test_mass.py
        python/test_units.py
    )

endif()

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(SofaPython SofaGTestMain)


# needed by python tests
AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")


include(${SOFA_CMAKE_DIR}/postProject.cmake)
