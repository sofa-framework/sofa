#include <SofaTest/Python_test.h>

namespace sofa {




// these are sofa scenes
static struct Tests : public Python_test_list {
    Tests() {
        addTestDir(COMPLIANT_TEST_PYTHON_DIR, "scene_");
        
    }
} tests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_scene_test,
                        ::testing::ValuesIn(tests.list));

TEST_P(Python_scene_test, sofa_python_scene_tests)
{
    max_steps *= 10; // scene_friction.py test requires tons of steps
    run(GetParam());
}




////////////////////////


// these are just python files loaded in the sofa python environment (paths...)
static struct Tests2 : public Python_test_list {
    Tests2() {
        addTestDir(COMPLIANT_TEST_PYTHON_DIR, "test_");
    }
} tests2;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_test,
                        ::testing::ValuesIn(tests2.list));

TEST_P(Python_test, sofa_python_tests)
{
    run(GetParam());
}




} // namespace sofa
