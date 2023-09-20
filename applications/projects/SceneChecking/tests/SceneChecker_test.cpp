/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/simulation/Node.h>

#include <SceneChecking/SceneCheckerVisitor.h>
using sofa::scenechecking::SceneCheckerVisitor;

#include <sofa/simulation/SceneCheck.h>
using sofa::simulation::SceneCheck;

#include <SceneChecking/SceneCheckAPIChange.h>
using sofa::scenechecking::SceneCheckAPIChange;
#include <SceneChecking/SceneCheckMissingRequiredPlugin.h>
using sofa::scenechecking::SceneCheckMissingRequiredPlugin;
#include <SceneChecking/SceneCheckDuplicatedName.h>
using sofa::scenechecking::SceneCheckDuplicatedName;
#include <SceneChecking/SceneCheckUsingAlias.h>
using sofa::scenechecking::SceneCheckUsingAlias;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML;
using sofa::simulation::Node;
using sofa::core::execparams::defaultInstance;

/////////////////////// COMPONENT DEFINITION & DECLARATION /////////////////////////////////////////
/// This component is only for testing the APIVersion system.
////////////////////////////////////////////////////////////////////////////////////////////////////
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::Base;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;
using sofa::core::ExecParams;

#include <sofa/simulation/graph/SimpleApi.h>

class ComponentDeprecated : public BaseObject
{
public:
    SOFA_CLASS(ComponentDeprecated, BaseObject);
public:

};

int ComponentDeprecatedClassId = sofa::core::RegisterObject("")
        .add< ComponentDeprecated >();


////////////////////////////////////// TEST ////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
struct SceneChecker_test : public BaseSimulationTest
{
    void SetUp() override
    {
    }

    void checkRequiredPlugin(bool missing)
    {
        PluginManager::getInstance().loadPluginByName("Sofa.Component.ODESolver.Forward");

        const std::string missStr = missing ? "" : "<RequiredPlugin name='Sofa.Component.ODESolver.Forward'/> \n";
        std::stringstream scene;
        scene << "<?xml version='1.0'?>                                             \n"
              << "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >      \n"
              << missStr
              << "      <EulerExplicitSolver />               \n"
              << "</Node>                                                           \n";

        SceneLoaderXML sceneLoader;
        const Node::SPtr root = sceneLoader.doLoadFromMemory ("testscene",
                                                              scene.str().c_str());
        EXPECT_MSG_NOEMIT(Error);

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        SceneCheckerVisitor checker(sofa::core::execparams::defaultInstance());
        checker.addCheck(SceneCheckMissingRequiredPlugin::newSPtr() );

        if(missing)
        {
            EXPECT_MSG_EMIT(Warning); // [SceneCheckMissingRequiredPlugin]
            checker.validate(root.get(), &sceneLoader);
        }
        else
        {
            EXPECT_MSG_NOEMIT(Warning);
            checker.validate(root.get(), &sceneLoader);
        }
    }

    void checkDuplicatedNames()
    {
        std::stringstream scene;
        scene << "<?xml version='1.0'?>                                           \n"
              << "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >    \n"
              << "    <RequiredPlugin name='Sofa.GL.Component'/>                   \n"
              << "    <Node name='nodeCheck'>                                     \n"
              << "      <Node name='nodeA' />                                     \n"
              << "      <Node name='nodeA' />                                     \n"
              << "    </Node>                                                     \n"
              << "    <Node name='objectCheck'>                                   \n"
              << "      <OglModel name='objectA' />                               \n"
              << "      <OglModel name='objectA' />                               \n"
              << "    </Node>                                                     \n"
              << "    <Node name='mixCheck'>                                      \n"
              << "      <Node name='mixA' />                                      \n"
              << "      <OglModel name='mixA' />                                  \n"
              << "    </Node>                                                     \n"
              << "    <Node name='nothingCheck'>                                  \n"
              << "      <Node name='nodeA' />                                     \n"
              << "      <OglModel name='objectA' />                               \n"
              << "    </Node>                                                     \n"
              << "</Node>                                                         \n";

        SceneLoaderXML sceneLoader;
        const Node::SPtr root = sceneLoader.doLoadFromMemory ("testscene",
                                                              scene.str().c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        SceneCheckerVisitor checker(sofa::core::execparams::defaultInstance());
        checker.addCheck( SceneCheckDuplicatedName::newSPtr() );

        const std::vector<std::string> nodenames = {"nodeCheck", "objectCheck", "mixCheck"};
        for( auto& nodename : nodenames )
        {
            EXPECT_MSG_NOEMIT(Error);
            EXPECT_MSG_EMIT(Warning);
            ASSERT_NE(root->getChild(nodename), nullptr);
            checker.validate(root->getChild(nodename), &sceneLoader);
        }

        {
            EXPECT_MSG_NOEMIT(Error);
            EXPECT_MSG_NOEMIT(Warning);
            ASSERT_NE(root->getChild("nothingCheck"), nullptr);
            checker.validate(root->getChild("nothingCheck"), &sceneLoader);
        }

    }

    void checkAPIVersion(bool shouldWarn)
    {
        EXPECT_MSG_NOEMIT(Error);
        EXPECT_MSG_NOEMIT(Warning);

        const std::string lvl = (shouldWarn)?"17.06":"17.12";

        std::stringstream scene;
        scene << "<?xml version='1.0'?>                                           \n"
              << "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >    \n"
              << "      <RequiredPlugin name='Sofa.Component.SceneUtility'/>      \n"
              << "      <APIVersion level='"<< lvl <<"'/>                         \n"
              << "      <ComponentDeprecated />                                   \n"
              << "</Node>                                                         \n";

        SceneLoaderXML sceneLoader;
        const Node::SPtr root = sceneLoader.doLoadFromMemory("testscene", scene.str().c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        SceneCheckerVisitor checker(sofa::core::execparams::defaultInstance());
        const SceneCheckAPIChange::SPtr apichange = SceneCheckAPIChange::newSPtr();
        apichange->installDefaultChangeSets();
        apichange->addHookInChangeSet("17.06", [](Base* o){
            if(o->getClassName() == "ComponentDeprecated")
                msg_warning(o) << "ComponentDeprecated have changed since 17.06.";
        });
        checker.addCheck(apichange);

        if(shouldWarn){
            /// We check that running a scene set to 17.12 generate a warning on a 17.06 component
            EXPECT_MSG_EMIT(Warning);
            checker.validate(root.get(), &sceneLoader);
        }
        else {
            checker.validate(root.get(), &sceneLoader);
        }
    }

    void checkUsingAlias(bool sceneWithAlias)
    {
        const std::string withAlias = "Mesh";
        const std::string withoutAlias = "MeshTopology";
        const std::string componentName = sceneWithAlias ? withAlias : withoutAlias;

        std::stringstream scene;
        scene << "<?xml version='1.0'?>                                           \n"
              << "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >    \n"
              << "    <RequiredPlugin name='Sofa.Component.StateContainer'/>      \n"
              << "    <RequiredPlugin name='Sofa.Component.Topology.Container.Constant'/>      \n"
              << "    <MechanicalObject template='Vec3d' />                       \n"
              << "    <" << componentName << "/>                                  \n"
              << "</Node>                                                         \n";


        SceneCheckerVisitor checker(sofa::core::execparams::defaultInstance());
        checker.addCheck( SceneCheckUsingAlias::newSPtr() );

        SceneLoaderXML sceneLoader;
        const Node::SPtr root = sceneLoader.doLoadFromMemory("testscene", scene.str().c_str());
        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        if(sceneWithAlias)
        {
            EXPECT_MSG_EMIT(Warning); // [SceneCheckUsingAlias]
            checker.validate(root.get(), &sceneLoader);
        }
        else
        {
            EXPECT_MSG_NOEMIT(Warning);
            checker.validate(root.get(), &sceneLoader);
        }
    }
};

TEST_F(SceneChecker_test, checkMissingRequiredPlugin )
{
    checkRequiredPlugin(true);
}

TEST_F(SceneChecker_test, checkPresentRequiredPlugin )
{
    checkRequiredPlugin(false);
}

TEST_F(SceneChecker_test, checkAPIVersion )
{
    checkAPIVersion(false);
}

TEST_F(SceneChecker_test, checkAPIVersionCurrent )
{
    checkAPIVersion(false);
}

TEST_F(SceneChecker_test, checkAPIVersionDeprecated )
{
    checkAPIVersion(true);
}

TEST_F(SceneChecker_test, checkDuplicatedNames )
{
    checkDuplicatedNames();
}

TEST_F(SceneChecker_test, checkUsingAlias_withAlias )
{
    checkUsingAlias(true);
}

TEST_F(SceneChecker_test, checkUsingAlias_withoutAlias )
{
    checkUsingAlias(false);
}
