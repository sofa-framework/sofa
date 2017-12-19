#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaGraphComponent/SceneCheckerVisitor.h>
using sofa::simulation::SceneCheckerVisitor ;

#include <SofaGraphComponent/SceneChecks.h>
using namespace sofa::simulation::scenecheckers ;

#include <SofaGraphComponent/SceneCheckAPIChanges.h>
using sofa::simulation::scenecheckers::SceneCheckAPIChange ;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

/////////////////////// COMPONENT DEFINITION & DECLARATION /////////////////////////////////////////
/// This component is only for testing the APIVersion system.
////////////////////////////////////////////////////////////////////////////////////////////////////
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;
using sofa::core::objectmodel::Base ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::ExecParams;
class ComponentDeprecated : public BaseObject
{
public:
    SOFA_CLASS(ComponentDeprecated, BaseObject) ;
public:

};

SOFA_DECL_CLASS(ComponentDeprecated)
int ComponentDeprecatedClassId = sofa::core::RegisterObject("")
        .add< ComponentDeprecated >() ;


////////////////////////////////////// TEST ////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
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

        SceneCheckerVisitor checker(ExecParams::defaultInstance());
        checker.addCheck( SceneCheckMissingRequiredPlugin::newSPtr() );

        if(missing)
        {
            EXPECT_MSG_EMIT(Warning);
            checker.validate(root.get()) ;
        }else{
            EXPECT_MSG_NOEMIT(Warning);
            checker.validate(root.get()) ;
        }
    }

    void checkDuplicatedNames()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
              << "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >                   \n"
              << "    <Node name='nodeCheck'>                                                    \n"
              << "      <Node name='nodeA' />                                                    \n"
              << "      <Node name='nodeA' />                                                    \n"
              << "    </Node>                                                                    \n"
              << "    <Node name='objectCheck'>                                                  \n"
              << "      <OglModel name='objectA' />                                              \n"
              << "      <OglModel name='objectA' />                                              \n"
              << "    </Node>                                                                    \n"
              << "    <Node name='mixCheck'>                                                     \n"
              << "      <Node name='mixA' />                                                     \n"
              << "      <OglModel name='mixA' />                                                 \n"
              << "    </Node>                                                                    \n"
              << "    <Node name='nothingCheck'>                                                 \n"
              << "      <Node name='nodeA' />                                                    \n"
              << "      <OglModel name='objectA' />                                              \n"
              << "    </Node>                                                                    \n"
              << "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        SceneCheckerVisitor checker(ExecParams::defaultInstance());
        checker.addCheck( SceneCheckDuplicatedName::newSPtr() );

        std::vector<std::string> nodenames = {"nodeCheck", "objectCheck", "mixCheck"} ;
        for( auto& nodename : nodenames )
        {
            EXPECT_MSG_NOEMIT(Error) ;
            EXPECT_MSG_EMIT(Warning);
            ASSERT_NE(root->getChild(nodename), nullptr) ;
            checker.validate(root->getChild(nodename)) ;
        }

        {
            EXPECT_MSG_NOEMIT(Error);
            EXPECT_MSG_NOEMIT(Warning);
            ASSERT_NE(root->getChild("nothingCheck"), nullptr) ;
            checker.validate(root->getChild("nothingCheck")) ;
        }

    }

    void checkAPIVersion(bool shouldWarn)
    {
        EXPECT_MSG_NOEMIT(Error) ;
        EXPECT_MSG_NOEMIT(Warning);

        std::string lvl = (shouldWarn)?"17.06":"17.12" ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
              << "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >                   \n"
              << "      <APIVersion level='"<< lvl <<"'/>                                        \n"
              << "      <ComponentDeprecated />                                                  \n"
              << "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        SceneCheckerVisitor checker(ExecParams::defaultInstance());
        SceneCheckAPIChange::SPtr apichange = SceneCheckAPIChange::newSPtr() ;
        apichange->installDefaultChangeSets() ;
        apichange->addHookInChangeSet("17.06", [](Base* o){
            if(o->getClassName() == "ComponentDeprecated")
                msg_warning(o) << "ComponentDeprecated have changed since 17.06." ;
        }) ;
        checker.addCheck(apichange) ;

        if(shouldWarn){
            /// We check that running a scene set to 17.12 generate a warning on a 17.06 component
            EXPECT_MSG_EMIT(Warning) ;
            checker.validate(root.get()) ;
        }
        else {
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

TEST_F(SceneChecker_test, checkAPIVersion )
{
    checkAPIVersion(false) ;
}

TEST_F(SceneChecker_test, checkAPIVersionCurrent )
{
    checkAPIVersion(false) ;
}

TEST_F(SceneChecker_test, checkAPIVersionDeprecated )
{
    checkAPIVersion(true) ;
}

TEST_F(SceneChecker_test, checkDuplicatedNames )
{
    checkDuplicatedNames() ;
}
