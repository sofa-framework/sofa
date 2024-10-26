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
#include <sofa/component/engine/select/MeshSubsetEngine.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/testing/BaseSimulationTest.h>

namespace sofa
{
template <typename _DataTypes>
struct MeshSubsetEngine_test : public testing::BaseSimulationTest
{
    sofa::simulation::Node::SPtr m_root;
    typename component::engine::select::MeshSubsetEngine<_DataTypes>::SPtr m_engine;

    void SetUp() override
    {
        simpleapi::importPlugin("Sofa.Component.Engine.Select");

        m_root = simulation::getSimulation()->createNewNode("root");
        ASSERT_NE(nullptr, m_root);

        m_engine = core::objectmodel::New<component::engine::select::MeshSubsetEngine<_DataTypes>>();
        ASSERT_NE(nullptr, m_engine);

        m_root->addObject(m_engine);
    }

    void TearDown() override
    {
        sofa::simulation::node::unload(m_root) ;
    }

    void testTwoTriangles()
    {
        m_engine->d_inputPosition.setValue({{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }});
        m_engine->d_inputTriangles.setValue({{ 0, 1, 2 }, { 0, 2, 3 }});
        m_engine->d_indices.setValue({0, 3, 2});

        m_engine->update();

        const auto& position = m_engine->d_position.getValue();
        EXPECT_EQ(position.size(), 3);

        const auto& inputPosition = m_engine->d_inputPosition.getValue();

        EXPECT_NE(std::find(position.begin(), position.end(), inputPosition[0]), position.end());
        EXPECT_EQ(std::find(position.begin(), position.end(), inputPosition[1]), position.end());
        EXPECT_NE(std::find(position.begin(), position.end(), inputPosition[2]), position.end());
        EXPECT_NE(std::find(position.begin(), position.end(), inputPosition[3]), position.end());

        const auto& triangles = m_engine->d_triangles.getValue();

        EXPECT_EQ(triangles.size(), 1);

        EXPECT_EQ(triangles[0][0], 0);
        EXPECT_EQ(triangles[0][1], 2);
        EXPECT_EQ(triangles[0][2], 1);
    }
};

using DataTypes = ::testing::Types<sofa::defaulttype::Vec3Types>;
TYPED_TEST_SUITE(MeshSubsetEngine_test, DataTypes);

TYPED_TEST(MeshSubsetEngine_test, testTwoTriangles)
{
    EXPECT_MSG_NOEMIT(Error);
    ASSERT_NO_THROW(this->testTwoTriangles());
}

}
