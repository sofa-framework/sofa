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
#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base ;
using sofa::core::objectmodel::ComponentState;

#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

#include <sofa/core/PathResolver.h>

using sofa::defaulttype::Rigid3Types;
using sofa::defaulttype::Vec3Types;

namespace
{

class BaseLink_simutest: public BaseSimulationTest,
        public ::testing::WithParamInterface<std::vector<std::string>>
{
public:
    SceneInstance* c;
    Node* node {nullptr};
    BaseLink_simutest()
    {
        importPlugin("SofaComponentAll") ;
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <MechanicalObject name='mstate0'/>                                      \n"
                 "   <InfoComponent name='obj'/>                                             \n"
                 "   <Node name='child1'>                                                    \n"
                 "      <MechanicalObject name='mstate1'/>                                   \n"
                 "      <Node name='child2'>                                                 \n"
                 "      </Node>                                                              \n"
                 "   </Node>                                                                 \n"
                 "</Node>                                                                    \n" ;
        c = new SceneInstance("xml", scene.str()) ;
        c->initScene() ;
        Node* root = c->root.get() ;
        Base* b = sofa::core::PathResolver::FindBaseFromPath(root, "@/child1/child2");
        node = dynamic_cast<Node*>(b);
    }

    ~BaseLink_simutest() override
    {
        delete c;
    }
};

TEST_F(BaseLink_simutest , testCheckpath )
{
    ASSERT_NE(node,nullptr);
    EXPECT_TRUE(node->object.CheckPaths("@../mstate1", node)) << "The mstate exists so this test should succeed";
    EXPECT_TRUE(node->object.CheckPaths("@../mstate1 @../../mstate0", node)) << "Unable to parse multiple link's target correctly";
    EXPECT_FALSE(node->object.CheckPaths("@../mstate", node)) << "The mstate does not exists so the returned value should be false";

    EXPECT_TRUE(node->object.CheckPaths("@/child1/mstate1", node)) << "The mstate exists so this test should succeed";
    EXPECT_TRUE(node->object.CheckPaths("@/child1/mstate1 @/mstate0", node)) << "Unable to parse multiple link's target correctly";
    EXPECT_TRUE(node->mechanicalState.CheckPath("@/child1/mstate1", node)) << "The pointed mstate exists and is of right type";
}

TEST_P(BaseLink_simutest, checkInvalidCheckPath)
{
    ASSERT_NE(node,nullptr);
    auto& t = GetParam();
    ASSERT_FALSE(node->object.CheckPaths(t[0], node)) << t[1];
}

std::vector<std::vector<std::string>> invalidValues={
    {"@/child1", "The types are different so this test should fails"},
    {"@/obj", "The types are different so this test should fails"},
    {"@/child1/mstate1 @/child1/mstate1", "A SingleLink should accept two links"}
};

INSTANTIATE_TEST_CASE_P(checkInvalidCheckPath,
                        BaseLink_simutest,
                        ::testing::ValuesIn(invalidValues));


} /// namespace
