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
#include <variant>
#include <sofa/simulation/SceneGraphObjectTraversal.h>
#include <gtest/gtest.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/simulation/graph/DAGNode.h>


namespace sofa
{

using simulation::graph::DAGNode;
using simulation::NodeIterator;
TEST(NodeIterator, constructor)
{
    {
        const simulation::NodeIterator<core::objectmodel::BaseObject> begin(nullptr);
        EXPECT_EQ(begin.ptr(), nullptr);
    }

    {
        const simulation::NodeIterator<sofa::core::behavior::BaseMechanicalState> begin(nullptr);
        EXPECT_EQ(begin.ptr(), nullptr);
    }

    const DAGNode::SPtr root = core::objectmodel::New<DAGNode>("root");
    {
        const simulation::NodeIterator<core::objectmodel::BaseObject> begin(root.get());
        EXPECT_EQ(begin.ptr(), nullptr);
    }
}

TEST(NodeIterator, oneElementInRoot)
{
    const DAGNode::SPtr root = core::objectmodel::New<DAGNode>("root");

    const auto dofs = core::objectmodel::New<component::statecontainer::MechanicalObject<defaulttype::Vec3Types>>();
    root->addObject(dofs);

    {
        simulation::NodeIterator<sofa::core::behavior::BaseMechanicalState> begin(root.get());
        EXPECT_EQ(begin.ptr(), dofs.get());

        begin++;
        EXPECT_EQ(begin.ptr(), nullptr);
    }

    {
        std::size_t counter {};
        for (const auto* state : simulation::SceneGraphObjectTraversal<sofa::core::behavior::BaseMechanicalState>(root.get()))
        {
            SOFA_UNUSED(state);
            ++counter;
        }

        EXPECT_EQ(counter, 1);
    }
}

TEST(NodeIterator, oneElementInRootAndOneElementInAChild)
{
    const DAGNode::SPtr root = core::objectmodel::New<DAGNode>("root");

    const auto dofs = core::objectmodel::New<component::statecontainer::MechanicalObject<defaulttype::Vec3Types>>();
    root->addObject(dofs);

    const DAGNode::SPtr child = core::objectmodel::New<DAGNode>("child");
    root->addChild(child);

    const auto dofsChild = core::objectmodel::New<component::statecontainer::MechanicalObject<defaulttype::Vec3Types>>();
    child->addObject(dofsChild);
    {
        std::size_t counter {};
        for (const auto* state : simulation::SceneGraphObjectTraversal<sofa::core::behavior::BaseMechanicalState>(root.get()))
        {
            SOFA_UNUSED(state);
            ++counter;
        }

        EXPECT_EQ(counter, 2);
    }
}

TEST(NodeIterator, oneElementInRootAndOneElementInEachChild)
{
    const DAGNode::SPtr root = core::objectmodel::New<DAGNode>("root");

    const auto dofs = core::objectmodel::New<component::statecontainer::MechanicalObject<defaulttype::Vec3Types>>();
    root->addObject(dofs);

    const DAGNode::SPtr child = core::objectmodel::New<DAGNode>("child");
    root->addChild(child);

    const auto dofsChild = core::objectmodel::New<component::statecontainer::MechanicalObject<defaulttype::Vec3Types>>();
    child->addObject(dofsChild);

    const DAGNode::SPtr child2 = core::objectmodel::New<DAGNode>("child2");
    child->addChild(child2);

    const auto dofsChild2 = core::objectmodel::New<component::statecontainer::MechanicalObject<defaulttype::Vec3Types>>();
    child2->addObject(dofsChild2);

    const DAGNode::SPtr child3 = core::objectmodel::New<DAGNode>("child3");
    root->addChild(child3);

    const auto dofsChild3 = core::objectmodel::New<component::statecontainer::MechanicalObject<defaulttype::Vec3Types>>();
    child3->addObject(dofsChild3);

    {
        std::size_t counter {};
        for (const auto* state : simulation::SceneGraphObjectTraversal<sofa::core::behavior::BaseMechanicalState>(root.get()))
        {
            SOFA_UNUSED(state);
            ++counter;
        }

        EXPECT_EQ(counter, 4);
    }
}

}
