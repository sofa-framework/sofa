#include <SofaTest/Python_test.h>

// TODO build a list of .py files by static
// initialization in a std::vector, then launch tests on them

// in the meantime, this:



const char* test_files[] = {
    ADD_PYTHON_TEST_DIR(COMPLIANT_TEST_PYTHON_DIR,"Example.py"),
    ADD_PYTHON_TEST_DIR(COMPLIANT_TEST_PYTHON_DIR,"LambdaPropagation.py"),
    ADD_PYTHON_TEST_DIR(COMPLIANT_TEST_PYTHON_DIR,"UniformCompliance.py"),
    ADD_PYTHON_TEST_DIR(COMPLIANT_TEST_PYTHON_DIR,"AffineMultiMapping.py")

	// add yours here
};

namespace sofa {

INSTANTIATE_TEST_CASE_P(Batch,
						Python_test,
						::testing::ValuesIn(test_files));

}
