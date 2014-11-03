#include <SofaTest/Python_test.h>


namespace sofa {


// static build of the test list
static struct Tests : public Python_test_list
{
    Tests()
    {
        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR);

        addTest( "test_Quaternion.py", scenePath );
        addTest( "test_Tools.py", scenePath );
        addTest( "test_units.py", scenePath );


        // add python tests here
    }
} tests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_test,
                        ::testing::ValuesIn(tests.list));

TEST_P(Python_test, sofa_python_test)
{
    run(GetParam());
}

}
