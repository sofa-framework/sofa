#include <SofaTest/Python_test.h>


namespace sofa {

//////////////////////////////////
////////// SOFA PYTHON SCENE TESTS
//////////////////////////////////

// static build of the test list
static struct SceneTests : public Python_test_list
{
    SceneTests()
    {
        static const std::string scenePath = std::string(SOFATEST_SCENES_DIR);

        // TODO create more test with several (random?) arguments

        std::vector<std::string> arguments(6);
        arguments[0] = "0.1"; // damping coef
        arguments[1] = "1.0"; // initial velocity
        arguments[2] = "0.01"; // dt
        arguments[3] = "1e-3"; // error threshold
        arguments[4] = "1.0"; // mass
        arguments[5] = "1.0"; // radius
        addTest( "damping.py", scenePath, arguments );

        // add python scene tests here
    }
} sceneTests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
						Python_scene_test,
                        ::testing::ValuesIn(sceneTests.list));

TEST_P(Python_scene_test, sofa_python_scene_tests)
{
    run(GetParam());
}


///////////


//////////////////////////////////
////////// PURE PYTHON TESTS
//////////////////////////////////

// static build of the test list
static struct Tests : public Python_test_list
{
    Tests()
    {
        static const std::string testPath = std::string(SOFATEST_PYTHONTESTS_DIR);

        std::vector<std::string> arguments(2);
        arguments[0] = "0"; // null
        arguments[1] = "1"; // not null
        addTest( "python_test_test.py", testPath, arguments );

        // add pure python tests here
    }
} tests;


// run test list
INSTANTIATE_TEST_CASE_P(Batch,
                        Python_test,
                        ::testing::ValuesIn(tests.list));

TEST_P(Python_test, sofa_python_tests)
{
    run(GetParam());
}

} // namespace sofa
