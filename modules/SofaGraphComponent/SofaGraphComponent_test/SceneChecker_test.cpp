#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaGraphComponent/SceneCheckerVisitor.h>
using sofa::simulation::SceneCheckerVisitor ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

using sofa::core::ExecParams;

struct SceneChecker_test : public Sofa_test<>
{
    void checkRequiredPlugin(bool missing)
    {
        EXPECT_MSG_EMIT(Error) ;
        EXPECT_MSG_NOEMIT(Warning);

        PluginManager::getInstance().loadPluginByName("SofaPython") ;

        std::string missStr = (missing)?"" : "<RequiredPlugin name='SofaPython'/> \n";
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
              << "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >                   \n"
              << missStr
              << "      <PythonScriptController classname='AClass' />                            \n"
              << "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        if(missing)
        {
            EXPECT_MSG_EMIT(Warning);
            SceneCheckerVisitor checker(ExecParams::defaultInstance());
            checker.validate(root.get()) ;
        }else{
            EXPECT_MSG_NOEMIT(Warning);
            SceneCheckerVisitor checker(ExecParams::defaultInstance());
            checker.validate(root.get()) ;
        }
    }
};

TEST_F(SceneChecker_test, checkMissingRequiredPlugin )
{
    checkRequiredPlugin(true) ;
}

TEST_F(SceneChecker_test, checkPresentRequiredPlugin )
{
    checkRequiredPlugin(false) ;
}
