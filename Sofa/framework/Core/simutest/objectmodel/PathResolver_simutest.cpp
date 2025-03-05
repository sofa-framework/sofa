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

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

#include <sofa/core/PathResolver.h>
using sofa::core::PathResolver;

#include <sofa/defaulttype/RigidTypes.h>
using sofa::defaulttype::Rigid3Types;

#include <sofa/defaulttype/VecTypes.h>
using sofa::defaulttype::Vec3Types;

namespace
{

class PathResolver_simutest : public BaseSimulationTest,
        public ::testing::WithParamInterface<std::vector<std::string>>
{
public:
    SceneInstance* c;
    Node* node {nullptr};
    PathResolver_simutest()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <DefaultAnimationLoop />                                                \n"
                 "   <DefaultVisualManagerLoop />                                            \n"
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
        const Node* root = c->root.get() ;
        Base* b = sofa::core::PathResolver::FindBaseFromPath(root, "@/child1/child2");
        node = dynamic_cast<Node*>(b);
    }

    ~PathResolver_simutest() override
    {
        delete c;
    }
};


class PathResolverToBaseObject : public PathResolver_simutest{};
TEST_P(PathResolverToBaseObject, CheckPathToBaseObject)
{
    ASSERT_NE(node,nullptr);
    auto& t = GetParam();
    if(t[2]=="true")
    {
        ASSERT_TRUE(PathResolver::CheckPath(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
        ASSERT_TRUE(PathResolver::CheckPaths(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
    }
    else
    {
        ASSERT_FALSE(PathResolver::CheckPath(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
        ASSERT_FALSE(PathResolver::CheckPaths(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
    }
}

std::vector<std::vector<std::string>> checkPathForBaseObjectValues=
{
    {"@/obj", "The linked type is an InfoComponent type while the link should point to a  object. CheckPath should return", "true"},
};

INSTANTIATE_TEST_SUITE_P(CheckPathToBaseObject,
                        PathResolverToBaseObject,
                        ::testing::ValuesIn(checkPathForBaseObjectValues));

// introduced in https://github.com/sofa-framework/sofa/pull/1717
TEST_F(PathResolverToBaseObject, DISABLED_CheckPathToBaseObject_tofix)
{
    const std::vector<std::string> t = { "@/child1", "The linked address is pointing to a node while trying to load it in an object. Using two different types should return", "false" };
    ASSERT_NE(node, nullptr);

    if (t[2] == "true")
    {
        ASSERT_TRUE(PathResolver::CheckPath(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
        ASSERT_TRUE(PathResolver::CheckPaths(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
    }
    else
    {
        ASSERT_FALSE(PathResolver::CheckPath(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
        ASSERT_FALSE(PathResolver::CheckPaths(node, BaseObject::GetClass(), t[0])) << t[1] << " " << t[2];
    }
}

class PathResolverToNode : public PathResolver_simutest{};
TEST_P(PathResolverToNode, CheckPathToNode)
{
    ASSERT_NE(node,nullptr);
    auto& t = GetParam();
    if(t[2]=="true")
    {
        ASSERT_TRUE(PathResolver::CheckPath(node, Node::GetClass(), t[0])) << t[1] << " " << t[2];
        ASSERT_TRUE(PathResolver::CheckPaths(node, Node::GetClass(), t[0])) << t[1] << " " << t[2];
    }
    else
    {
        ASSERT_FALSE(PathResolver::CheckPath(node, Node::GetClass(), t[0])) << t[1] << " " << t[2];
        ASSERT_FALSE(PathResolver::CheckPaths(node, Node::GetClass(), t[0])) << t[1] << " " << t[2];
    }
}

std::vector<std::vector<std::string>> checkPathToNodeValues=
{
    {"@/child1", "The linked address is pointing to a node, searching for a node so should return", "true"},
    {"@/obj", "The linked type is an InfoComponent type while the link should point to a node. CheckPath should return", "false"},
};

INSTANTIATE_TEST_SUITE_P(CheckPathToNode,
                        PathResolverToNode,
                        ::testing::ValuesIn(checkPathToNodeValues));

} /// namespace
