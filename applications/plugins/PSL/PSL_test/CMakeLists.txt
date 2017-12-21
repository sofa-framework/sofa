cmake_minimum_required(VERSION 3.1)

project(PSL_test)


set(HEADER_FILES)

set(SOURCE_FILES
        PSL_test.cpp
        python_test.cpp
        )

set(PSL_FILES
        pslfiles/test_syntax.psl
        pslfiles/test_syntax_python.psl
        pslfiles/test_syntax_object.psl
        pslfiles/test_emptyfile.psl
        pslfiles/test_emptyfile.pslx
        pslfiles/test_brokenfile.psl
        pslfiles/test_brokenfile.pslx
        pslfiles/test_node.psl
        pslfiles/test_node.pslx
        pslfiles/test_node_fail2.psl
        pslfiles/test_node.psl
        pslfiles/test_properties.psl
        pslfiles/test_properties_raw.psl
        pslfiles/test_pslversion.psl
        pslfiles/test_pythonglobals.psl
        pslfiles/test_pythonlocals.psl
        pslfiles/test_template.psl
        pslfiles/test_template_dynamic.psl
        pslfiles/test_template_frame.psl
        pslfiles/test_template_kwargs.psl
        pslfiles/test_template_empty.psl
        pslfiles/test_node_fail.psl
        pslfiles/test_object.psl
        pslfiles/test_object_datafield.psl
        pslfiles/test_pslversion_invalid.psl
        pslfiles/test_pythonexcept.psl
        pslfiles/test_pythonobject.psl
        pslfiles/test_python.psl
        pslfiles/test_pythondsl.psl
        pslfiles/test_import.psl
        pslfiles/test_importpython.psl
        pslfiles/test_importedpythondsl.psl
        pslfiles/importedlib.psl
        pslfiles/importedpython.py
        pslfiles/importedpythondsl.py
)

set(PYTHON_TEST_FILES
   pyfiles/pslloader_test.py
   pyfiles/data/test_ast.psl
   pyfiles/data/test_ast.pslx
   pyfiles/data/test_ast_broken.pslx
   pyfiles/data/test_caduceus.psl
   pyfiles/data/test_caduceus.pslx
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PSL_FILES} ${PYTHON_TEST_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest PSL)
add_definitions("-DPSL_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pslfiles/\"" "-DPYTHON_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/pyfiles/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
