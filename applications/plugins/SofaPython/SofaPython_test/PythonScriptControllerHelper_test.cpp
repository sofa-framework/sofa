#include <gtest/gtest.h>

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

#include <SofaPython/PythonScriptControllerHelper.h>

using namespace sofa::helper;

namespace sofa {

struct PythonScriptHelper_test : public ::testing::Test
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

    void checkResult()
    {
        bool b;
        PythonScriptController_call(b, "controller", "getTrue");
        EXPECT_TRUE(b);
        PythonScriptController_call(b, "controller", "getFalse");
        EXPECT_FALSE(b);
        int i;
        PythonScriptController_call(i, "controller", "getInt");
        EXPECT_EQ(7, i);
        double d;
        PythonScriptController_call(d, "controller", "getFloat");
        EXPECT_FLOAT_EQ(12.34, d);
        float f;
        PythonScriptController_call(f, "controller", "getFloat");
        EXPECT_FLOAT_EQ(12.34, f);
        std::string s;
        PythonScriptController_call(s, "controller", "getString");
        EXPECT_EQ("test string", s);
    }

};

TEST_F(PythonScriptHelper_test, result)
{
    checkResult();
}

}
