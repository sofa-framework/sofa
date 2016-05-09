#include <SofaTest/Python_test.h>


namespace sofa {


// static build of the test list
static struct Tests : public Python_test_list
{
    Tests()
    {
        static const std::string scenePath = std::string(SOFABASEMECHANICS_TEST_PYTHON_DIR);

        addTest( "MechanicalObject_test.py", scenePath );
    }
} tests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_scene_test,
                        ::testing::ValuesIn(tests.list));

TEST_P(Python_scene_test, sofa_python_scene_tests)
{
    run(GetParam());
}




} // namespace sofa
