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
#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/component/sceneutility/InfoComponent.h>
using sofa::component::sceneutility::InfoComponent;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager ;

#include <sofa/simpleapi/SimpleApi.h>

class NodeContext_test: public BaseSimulationTest
{
public:


    NodeContext_test()
    {
        sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);
        sofa::simpleapi::importPlugin(Sofa.Component.SceneUtility);
    }

    void testGetNodeObjects()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <InfoComponent/>                                                             \n"
                 "   <Node name='child1'>                                                    \n"
                 "      <InfoComponent/>                                                          \n"
                 "      <InfoComponent/>                                                          \n"
                 "      <MechanicalObject />                                                 \n"
                 "      <Node name='child2'>                                                 \n"
                 "          <InfoComponent/>                                                      \n"
                 "          <InfoComponent/>                                                      \n"
                 "      </Node>                                                              \n"
                 "   </Node>                                                                 \n"
                 "</Node>                                                                    \n" ;

        SceneInstance c("xml", scene.str()) ;
        c.initScene() ;

        Node* m_root = c.root.get() ;
        ASSERT_NE(m_root, nullptr) ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        Node* node =m_root->getChild("child1") ;

        /// Query a specific model in a container, this is the old API
        std::vector<InfoComponent*> results ;
        node->getNodeObjects<InfoComponent, std::vector<InfoComponent*> >( &results ) ;
        ASSERT_EQ( results.size() , (unsigned int)2 ) ;

        /// Query a specific model with a nicer syntax
        std::vector<InfoComponent*> results2 ;
        ASSERT_EQ( node->getNodeObjects(results2).size(), (unsigned int)2 ) ;

        /// Query a specific model with a nicer syntax
        std::vector<InfoComponent*> results3 ;
        ASSERT_EQ( node->getNodeObjects(&results3)->size(), (unsigned int)2 ) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        /// So there is 4 base object in the scene.
        for(const auto& m : node->getNodeObjects() ) { SOFA_UNUSED(m); }
        ASSERT_EQ( node->getNodeObjects().size(), (unsigned int)3 ) ;
    }

    void testGetTreeObjects()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <InfoComponent/>                                                             \n"
                 "   <Node name='child1'>                                                    \n"
                 "      <InfoComponent/>                                                          \n"
                 "      <InfoComponent/>                                                          \n"
                 "      <MechanicalObject />                                                 \n"
                 "      <Node name='child2'>                                                 \n"
                 "          <InfoComponent/>                                                      \n"
                 "          <InfoComponent/>                                                      \n"
                 "      </Node>                                                              \n"
                 "   </Node>                                                                 \n"
                 "</Node>                                                                    \n" ;

        SceneInstance c("xml", scene.str()) ;
        c.initScene() ;

        Node* m_root = c.root.get() ;
        ASSERT_NE(m_root, nullptr) ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        Node* node =m_root->getChild("child1") ;

        /// Query a specific model in a container, this is the old API
        std::vector<InfoComponent*> results ;
        node->getTreeObjects<InfoComponent, std::vector<InfoComponent*> >( &results ) ;
        ASSERT_EQ( results.size() , (unsigned int)4  ) ;

        /// Query a specific model with a nicer syntax
        std::vector<InfoComponent*> results2 ;
        ASSERT_EQ( node->getTreeObjects(results2).size(), (unsigned int)4 ) ;

        /// Query a specific model with a nicer syntax
        std::vector<InfoComponent*> results3 ;
        ASSERT_EQ( node->getTreeObjects(&results3)->size(), (unsigned int)4 ) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        /// So there is 4 base object in the scene.
        for(const auto& m : node->getTreeObjects() ) { SOFA_UNUSED(m); }
        ASSERT_EQ( node->getTreeObjects().size(), (unsigned int)5 ) ;
    }
};

TEST_F(NodeContext_test , testGetNodeObjects )
{
    this->testGetNodeObjects();
}

TEST_F(NodeContext_test , testGetTreeObjects )
{
    this->testGetTreeObjects();
}
