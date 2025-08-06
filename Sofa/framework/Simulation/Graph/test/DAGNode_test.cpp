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
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/simulation/graph/Node.h>

using namespace sofa;
using namespace simulation::graph;

struct Node_test : public BaseTest
{
    Node_test() {}

    void test_findCommonParent()
    {
        const Node::SPtr root = core::objectmodel::New<Node>("root");
        const Node::SPtr node1 = core::objectmodel::New<Node>("node1");
        const Node::SPtr node2 = core::objectmodel::New<Node>("node2");
        const Node::SPtr node3 = core::objectmodel::New<Node>("node3");
        const Node::SPtr node11 = core::objectmodel::New<Node>("node11");
        const Node::SPtr node12 = core::objectmodel::New<Node>("node12");
        const Node::SPtr node31 = core::objectmodel::New<Node>("node31");

        root->addChild(node1);
        root->addChild(node2);
        root->addChild(node3);

        node1->addChild(node11);
        node1->addChild(node12);

        node3->addChild(node31);

        const simulation::Node* commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node11.get()) );
        EXPECT_STREQ(node1->getName().c_str(), commonParent->getName().c_str());

        commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node31.get()));
        EXPECT_STREQ(root->getName().c_str(), commonParent->getName().c_str());

        commonParent = node12->findCommonParent(static_cast<simulation::Node*>(node1.get()));
        EXPECT_STREQ(root->getName().c_str(), commonParent->getName().c_str());
    }

    void test_findCommonParent_MultipleParents()
    {
        const Node::SPtr root = core::objectmodel::New<Node>("root");
        const Node::SPtr node1 = core::objectmodel::New<Node>("node1");
        const Node::SPtr node2 = core::objectmodel::New<Node>("node2");
        const Node::SPtr node11 = core::objectmodel::New<Node>("node11");
        const Node::SPtr node22 = core::objectmodel::New<Node>("node22");
        const Node::SPtr node23 = core::objectmodel::New<Node>("node23");

        root->addChild(node1);
        root->addChild(node2);

        node1->addChild(node11);
        node1->addChild(node22);

        node2->addChild(node11);
        node2->addChild(node22);  
        node2->addChild(node23);

        const simulation::Node* commonParent = node11->findCommonParent(static_cast<simulation::Node*>(node22.get()));

        bool result = false;
        if (commonParent->getName().compare(node1->getName()) == 0 || commonParent->getName().compare(node2->getName()) == 0)
        {
            result = true;
        }
        EXPECT_TRUE(result);

        commonParent = node11->findCommonParent(static_cast<simulation::Node*>(node23.get()));
        EXPECT_STREQ(node2->getName().c_str(), commonParent->getName().c_str());
    }
};

TEST_F(Node_test, test_findCommonParent) { test_findCommonParent(); }
TEST_F(Node_test, test_findCommonParent_MultipleParents) { test_findCommonParent_MultipleParents(); }
