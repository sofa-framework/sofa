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

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

#include <sofa/core/objectmodel/DDGNode.h>
using sofa::core::objectmodel::DDGNode;

class DDGNodeTestClass : public DDGNode
{
public:
    int m_cpt {0};
    int m_cptNotify {0};

    void update() override {
        m_cpt++;
    }
    void notifyEndEdit() override
    {
        m_cptNotify++;
    }
};

class DDGNode_test: public BaseTest
{
public:
    DDGNodeTestClass m_ddgnode1;
    DDGNodeTestClass m_ddgnode2;
    DDGNodeTestClass m_ddgnode3;
};

TEST_F(DDGNode_test, addInput)
{
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), size_t(0));
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), size_t(0));
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), size_t(0));

    m_ddgnode1.addInput(&m_ddgnode2);
    EXPECT_EQ(m_ddgnode1.getInputs().size(), size_t(1));
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), size_t(1));

    m_ddgnode1.addInput(&m_ddgnode3);
    EXPECT_EQ(m_ddgnode1.getInputs().size(), size_t(2));
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), size_t(1));
}

TEST_F(DDGNode_test, addOutput)
{
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), size_t(0));
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), size_t(0));
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), size_t(0));

    m_ddgnode1.addOutput(&m_ddgnode2);
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), size_t(1));
    EXPECT_EQ(m_ddgnode2.getInputs().size(), size_t(1));

    m_ddgnode1.addOutput(&m_ddgnode3);
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), size_t(2));
    EXPECT_EQ(m_ddgnode3.getInputs().size(), size_t(1));
}

TEST_F(DDGNode_test, dellInput)
{
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), 0);

    m_ddgnode1.addInput(&m_ddgnode2);
    m_ddgnode1.addInput(&m_ddgnode3);

    m_ddgnode1.delInput(&m_ddgnode2);
    m_ddgnode1.delInput(&m_ddgnode3);

    EXPECT_EQ(m_ddgnode1.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), 0);
}

TEST_F(DDGNode_test, dellOutput)
{
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), 0);

    m_ddgnode1.addInput(&m_ddgnode2);
    m_ddgnode1.addInput(&m_ddgnode3);

    m_ddgnode2.delOutput(&m_ddgnode1);
    m_ddgnode3.delOutput(&m_ddgnode1);

    EXPECT_EQ(m_ddgnode1.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), 0);
}

TEST_F(DDGNode_test, propagationScenario1)
{
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), 0);

    m_ddgnode1.addInput(&m_ddgnode2);
    m_ddgnode1.addInput(&m_ddgnode3);

    m_ddgnode2.setDirtyOutputs();

    EXPECT_TRUE(m_ddgnode1.isDirty());
    EXPECT_FALSE(m_ddgnode2.isDirty());
    EXPECT_FALSE(m_ddgnode3.isDirty());
}

TEST_F(DDGNode_test, propagationScenario2)
{
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), 0);

    m_ddgnode1.m_cpt = 0;
    m_ddgnode2.m_cpt = 0;
    m_ddgnode3.m_cpt = 0;

    m_ddgnode1.addInput(&m_ddgnode2);
    m_ddgnode2.addInput(&m_ddgnode3);

    m_ddgnode3.setDirtyOutputs();

    EXPECT_TRUE(m_ddgnode1.isDirty());
    EXPECT_TRUE(m_ddgnode2.isDirty());
    EXPECT_FALSE(m_ddgnode3.isDirty());
    EXPECT_EQ(m_ddgnode1.m_cpt, 0);
    EXPECT_EQ(m_ddgnode2.m_cpt, 0);

    m_ddgnode1.updateIfDirty();
    m_ddgnode2.updateIfDirty();
    EXPECT_EQ(m_ddgnode1.m_cpt, 1);
    EXPECT_EQ(m_ddgnode2.m_cpt, 1);
}

TEST_F(DDGNode_test, propagationScenario3)
{
    EXPECT_EQ(m_ddgnode1.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode2.getOutputs().size(), 0);
    EXPECT_EQ(m_ddgnode3.getOutputs().size(), 0);

    m_ddgnode1.m_cpt = 0;
    m_ddgnode2.m_cpt = 0;
    m_ddgnode3.m_cpt = 0;

    m_ddgnode1.addInput(&m_ddgnode2);
    m_ddgnode2.addInput(&m_ddgnode3);

    m_ddgnode3.setDirtyOutputs();
    m_ddgnode3.notifyEndEdit();

    EXPECT_TRUE(m_ddgnode1.isDirty());
    EXPECT_TRUE(m_ddgnode2.isDirty());
    EXPECT_FALSE(m_ddgnode3.isDirty());
    EXPECT_EQ(m_ddgnode1.m_cpt, 0);
    EXPECT_EQ(m_ddgnode2.m_cpt, 0);

    m_ddgnode1.updateIfDirty();
    m_ddgnode2.updateIfDirty();
    EXPECT_EQ(m_ddgnode1.m_cpt, 1);
    EXPECT_EQ(m_ddgnode2.m_cpt, 1);
}
