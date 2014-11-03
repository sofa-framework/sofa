#include <SofaTest/Python_test.h>


namespace sofa {


// static build of the test list
static struct Tests : public Python_test_list
{
    Tests()
    {
        static const std::string scenePath = std::string(COMPLIANT_TEST_PYTHON_DIR);

        addTest( "Example.py", scenePath );
        addTest( "LambdaPropagation.py", scenePath );
        addTest( "UniformCompliance.py", scenePath );
        addTest( "AffineMultiMapping.py", scenePath );
        addTest( "restitution.py", scenePath );
        addTest( "friction.py", scenePath );

        // add python tests here
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




////////////////////////


// static build of the test list
static struct Tests2 : public Python_test_list
{
    Tests2()
    {
        static const std::string testPath = std::string(COMPLIANT_TEST_PYTHON_DIR);

        addTest( "GenerateRigid.py", testPath );

        // add pure python tests here
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
