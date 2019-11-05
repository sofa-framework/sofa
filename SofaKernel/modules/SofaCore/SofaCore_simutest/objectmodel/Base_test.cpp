/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
/******************************************************************************
 * Contributors:
 *     - damien.marchal@univ-lille.fr
 *****************************************************************************/

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base ;
using sofa::core::objectmodel::ComponentState;

#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

class Base_test: public BaseSimulationTest
{
public:
    virtual ~Base_test(){}
    void testComponentState()
    {
        EXPECT_MSG_NOEMIT(Error, Warning) ;
        importPlugin("SofaComponentAll") ;
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <Node name='child1'>                                                    \n"
                 "      <MechanicalObject />                                                 \n"
                 "      <Node name='child2'>                                                 \n"
                 "      </Node>                                                              \n"
                 "   </Node>                                                                 \n"
                 "</Node>                                                                    \n" ;

        SceneInstance c("xml", scene.str()) ;
        c.initScene() ;

        Node* root = c.root.get() ;
        ASSERT_NE(root, nullptr) ;

        ASSERT_NE(root->findData("componentState"), nullptr);
        root->m_componentstate = ComponentState::Valid;
        ASSERT_EQ(root->m_componentstate, ComponentState::Valid);
        root->m_componentstate = ComponentState::Loading;
        ASSERT_EQ(root->m_componentstate, ComponentState::Loading);
    }
};

TEST_F(Base_test , testComponentState )
{
    this->testComponentState();
}

