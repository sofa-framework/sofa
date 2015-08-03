include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    Regression_test.h
)

set(SOURCE_FILES
   Regression_test.cpp
        )


if(SOFA-PLUGIN_SOFAPYTHON)
    AddCompilerDefinitions("SOFATEST_PYTHONTESTS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python_tests\"")
    AddCompilerDefinitions("SOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    AddLinkerDependencies(SofaPython)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})


AddLinkerDependencies(SofaTest SofaPython SofaGTestMain)

AddCompilerDefinitions("SOFA_SRC_DIR=\"${SOFA_SRC_DIR}\"")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
