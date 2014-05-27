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

        // add python tests here
    }
} tests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_scene_test,
                        ::testing::ValuesIn(tests.list));

}
