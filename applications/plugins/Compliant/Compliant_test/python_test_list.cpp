#include <SofaTest/Python_test.h>


namespace sofa {


struct Tests : public Python_test_list
{
    Tests()
    {
        const std::string scenePath = std::string(COMPLIANT_TEST_PYTHON_DIR);

        addTest( "Example.py", scenePath );
        addTest( "LambdaPropagation.py", scenePath );
        addTest( "UniformCompliance.py", scenePath );
        addTest( "AffineMultiMapping.py", scenePath );

        // add python tests here
    }
};


// static build of the test list
static const Tests tests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_test,
                        ::testing::ValuesIn(tests.list));

}
