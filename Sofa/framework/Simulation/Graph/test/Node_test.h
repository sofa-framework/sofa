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
#include <gtest/gtest.h>

#include <sofa/core/objectmodel/BaseNode.h>

class Dummy: public sofa::core::objectmodel::BaseObject
{
    bool *m_destroyed;
public:
    SOFA_CLASS(Dummy, sofa::core::objectmodel::BaseObject);

    Dummy(const std::string& name): m_destroyed(nullptr) {this->setName(name);}
    Dummy(bool *destroyed): m_destroyed(destroyed) {}
    ~Dummy() override { if(m_destroyed) *m_destroyed = true; }
};

template<class Node>
void Node_test_objectDestruction_singleObject()
{
    const sofa::core::objectmodel::BaseNode::SPtr node = sofa::core::objectmodel::New<Node>();
    bool objectDestroyed = false;
    {
        const Dummy::SPtr dummy = sofa::core::objectmodel::New<Dummy>(&objectDestroyed);
        node->addObject(dummy);
        node->removeObject(dummy);
    }
    EXPECT_TRUE(objectDestroyed);
}

template<class Node>
void Node_test_objectDestruction_multipleObjects()
{
    const sofa::core::objectmodel::BaseNode::SPtr node = sofa::core::objectmodel::New<Node>();
    bool objectDestroyed[4] = {false};
    {
        Dummy::SPtr dummy[4];
        for (int i=0 ; i<4 ; i++)
        {
            dummy[i] = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[i]);
            node->addObject(dummy[i]);
        }
        node->removeObject(dummy[3]);
        node->removeObject(dummy[0]);
        node->removeObject(dummy[2]);
        node->removeObject(dummy[1]);
    }
    EXPECT_TRUE(objectDestroyed[0]);
    EXPECT_TRUE(objectDestroyed[1]);
    EXPECT_TRUE(objectDestroyed[2]);
    EXPECT_TRUE(objectDestroyed[3]);
}

template<class Node>
void Node_test_objectDestruction_childNode_singleObject()
{
    const sofa::core::objectmodel::BaseNode::SPtr node = sofa::core::objectmodel::New<Node>();
    bool objectDestroyed = false;
    {
        const sofa::core::objectmodel::BaseNode::SPtr childNode = sofa::core::objectmodel::New<Node>();
        node->addChild(childNode);
        const Dummy::SPtr dummy = sofa::core::objectmodel::New<Dummy>(&objectDestroyed);
        childNode->addObject(dummy);
        node->removeChild(childNode);
    }
    EXPECT_TRUE(objectDestroyed);
}

template<class Node>
void Node_test_objectDestruction_childNode_complexChild()
{
    bool objectDestroyed[10] = {false};
    const sofa::core::objectmodel::BaseNode::SPtr node = sofa::core::objectmodel::New<Node>();
    {
        const sofa::core::objectmodel::BaseNode::SPtr childNode1 = sofa::core::objectmodel::New<Node>();
        node->addChild(childNode1);
        const sofa::core::objectmodel::BaseNode::SPtr childNode2 = sofa::core::objectmodel::New<Node>();
        childNode1->addChild(childNode2);
        const sofa::core::objectmodel::BaseNode::SPtr childNode3 = sofa::core::objectmodel::New<Node>();
        childNode2->addChild(childNode3);
        const sofa::core::objectmodel::BaseNode::SPtr childNode4 = sofa::core::objectmodel::New<Node>();
        childNode1->addChild(childNode4);

        const Dummy::SPtr dummy1 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[0]);
        childNode1->addObject(dummy1);
        const Dummy::SPtr dummy2 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[1]);
        childNode1->addObject(dummy2);
        const Dummy::SPtr dummy3 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[2]);
        childNode2->addObject(dummy3);
        const Dummy::SPtr dummy4 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[3]);
        childNode2->addObject(dummy4);
        const Dummy::SPtr dummy5 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[4]);
        childNode3->addObject(dummy5);
        const Dummy::SPtr dummy6 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[5]);
        childNode3->addObject(dummy6);
        const Dummy::SPtr dummy7 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[6]);
        childNode4->addObject(dummy7);
        const Dummy::SPtr dummy8 = sofa::core::objectmodel::New<Dummy>(&objectDestroyed[7]);
        childNode4->addObject(dummy8);

        node->removeChild(childNode1);
    }
    EXPECT_TRUE(objectDestroyed[0]);
    EXPECT_TRUE(objectDestroyed[1]);
    EXPECT_TRUE(objectDestroyed[2]);
    EXPECT_TRUE(objectDestroyed[3]);
    EXPECT_TRUE(objectDestroyed[4]);
    EXPECT_TRUE(objectDestroyed[5]);
    EXPECT_TRUE(objectDestroyed[6]);
    EXPECT_TRUE(objectDestroyed[7]);
}
