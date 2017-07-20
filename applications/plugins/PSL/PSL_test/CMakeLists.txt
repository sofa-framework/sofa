cmake_minimum_required(VERSION 3.1)

project(PSL_test)


set(HEADER_FILES)

set(SOURCE_FILES
        PSL_test.cpp
        )

set(PSL_FILES
        pslfiles/test_node.psl
        pslfiles/test_node_fail2.psl
        pslfiles/test_node.psl
        pslfiles/test_properties.psl
        pslfiles/test_pslversion.psl
        pslfiles/test_pythonglobals.psl
        pslfiles/test_pythonlocals.psl
        pslfiles/test_template.psl
        pslfiles/test_node_fail.psl
        pslfiles/test_object.psl
        pslfiles/test_pslversion_invalid.psl
        pslfiles/test_pythonexcept.psl
        pslfiles/test_pythonobject.psl
        pslfiles/test_python.psl
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PSL_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest PSL)
add_definitions("-DPSL_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pslfiles/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
