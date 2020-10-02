#include <SofaTest/Python_test.h>


namespace sofa {

/////////////////////////
//// PURE PYTHON TESTS
/////////////////////////


// static build of the test list
static struct Tests : public Python_test_list
{
    Tests()
    {
        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR);

        addTest( "test_Quaternion.py", scenePath );
        addTest( "test_Tools.py", scenePath );
        addTest( "test_units.py", scenePath );
        addTest( "test_mass.py", scenePath );
        addTest( "test_MeshLoader.py", scenePath );

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


///////////////


/////////////////////////
//// PYTHON SCENE TESTS
/////////////////////////

// static build of the test list
static struct SceneTests : public Python_test_list
{
    SceneTests()
    {
        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR);

        // TODO create more test with several (random?) arguments

        addTest( "sysPathDuplicate.py", scenePath );
        addTest( "dataVecResize.py", scenePath );
        addTest( "automaticNodeInitialization.py", scenePath );
        addTest( "unicodeData.py", scenePath);
        
        // call it several times in the same python environment to simulate a reload
        for( int i=0 ; i<5 ; ++i )
            addTest( "moduleReload.py",  scenePath );

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

}
