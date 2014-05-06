#include <SofaTest/Python_test.h>


const char* test_files[] = {
    ADD_PYTHON_TEST_DIR(SOFATEST_SCENES_DIR,"damping.py")

    // add python tests here
};

namespace sofa {

INSTANTIATE_TEST_CASE_P(Batch,
						Python_test,
						::testing::ValuesIn(test_files));

}
