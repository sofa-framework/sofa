#include <gtest/gtest.h>

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

#include <SofaPython/PythonScriptControllerHelper.h>

using namespace sofa::helper;

namespace sofa {

struct PythonScriptControllerHelper_test : public ::testing::Test
{
protected:

    virtual void SetUp()
    {
        static const std::string scenePath = std::string(SOFAPYTHON_TEST_PYTHON_DIR)+std::string("/test_PythonScriptControllerHelper.py");

        // sofa init
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());
        // load scene
        simulation::Node::SPtr root =  simulation::getSimulation()->load(scenePath.c_str());
        simulation::getSimulation()->init(root.get());
    }

    void testResult()
    {
        bool b;
        PythonScriptController_call(&b, "controller", "getTrue");
        EXPECT_TRUE(b);
        PythonScriptController_call(&b, "controller", "getFalse");
        EXPECT_FALSE(b);
        int i;
        PythonScriptController_call(&i, "controller", "getInt");
        EXPECT_EQ(7, i);
        double d;
        PythonScriptController_call(&d, "controller", "getFloat");
        EXPECT_FLOAT_EQ(12.34, d);
        float f;
        PythonScriptController_call(&f, "controller", "getFloat");
        EXPECT_FLOAT_EQ(12.34, f);
        std::string s;
        PythonScriptController_call(&s, "controller", "getString");
        EXPECT_EQ("test string", s);
        int* none=nullptr;
        PythonScriptController_call(none, "controller", "getNone"); // just for testing
        EXPECT_EQ(nullptr, none);
        PythonScriptController_call(nullptr, "controller", "getNone"); // this is the way you do in your code
        PythonScriptController_call(none, "controller", "getNothing"); // just for testing
        EXPECT_EQ(nullptr, none);
        PythonScriptController_call(nullptr, "controller", "getNothing"); // this is the way you do in your code
    }

    void testParam()
    {
        int i;
        PythonScriptController_call(&i, "controller", "add", -5, 6);
        EXPECT_EQ(1, i);

        unsigned int ui;
        PythonScriptController_call(&ui, "controller", "add", (unsigned int)5, (unsigned int)6);
        EXPECT_EQ(11u, ui);

        double d;
        PythonScriptController_call(&d, "controller", "add", 7., 8.5);
        EXPECT_FLOAT_EQ(15.5, d);

        std::string s;
        PythonScriptController_call(&s, "controller", "add", (std::string)"hello ", (std::string)"world !"); // the cast looks necessary...
        EXPECT_EQ("hello world !", s);
    }

};

TEST_F(PythonScriptControllerHelper_test, result)
{
    testResult();
}

TEST_F(PythonScriptControllerHelper_test, param)
{
    testParam();
}

}
