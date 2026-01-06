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
#include <sofa/core/topology/BaseTopologyData.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <gtest/gtest.h>

namespace sofa::core::topology
{

class SimplePointTopology: public BaseMeshTopology
{
   public:
    SimplePointTopology(unsigned size)
    : m_data(initData(&m_data,"topologoSI",""))
    {
        m_points.reserve(size);
        for(unsigned i=0; i<size; ++i)
        {
            m_points.push_back(i);
        }
    }

    virtual void init()
    {
        BaseMeshTopology::init();
        m_data.createTopologyHandler(this);
    }

    const SeqEdges& getEdges() override { return m_edges; }
    const SeqTriangles& getTriangles() override { return m_triangles; }
    const SeqQuads& getQuads() override { return m_quads; }
    const SeqTetrahedra& getTetrahedra() override { return m_tetra; }
    const SeqHexahedra& getHexahedra() override { return m_hexa; }
    const SeqPrisms& getPrisms() override { return m_prisms; }
    const SeqPyramids& getPyramids() override { return m_pyramids; }

    virtual sofa::geometry::ElementType getTopologyType() const
    {
        return sofa::geometry::ElementType::POINT;
    }


    virtual Size getNbPoints() const { return m_points.size(); }
    void removePoints(const unsigned nb)
    {
        if(nb >= m_points.size())
            m_points.clear();
        else
            m_points.resize(m_points.size() - nb);
    }

    void addPoints(const unsigned nb)
    {
        m_points.reserve(m_points.size() + nb);
        for(unsigned i=m_points.size(); i<(m_points.size() + nb);++i)
            m_points.push_back(i);
    }

    sofa::type::vector<Edge> m_edges;
    sofa::type::vector<Triangle> m_triangles;
    sofa::type::vector<Quad> m_quads;
    sofa::type::vector<Tetra> m_tetra;
    sofa::type::vector<Hexa> m_hexa;
    sofa::type::vector<Prism> m_prisms;
    sofa::type::vector<Pyramid> m_pyramids;

    TopologySubsetIndices m_data;
    type::vector<sofa::Index> m_points;
};

TEST(TopologySubsetIndices_test, removePoints)
{
    SimplePointTopology PointContainer(4);
    PointContainer.init();
    PointContainer.m_data.setValue({0,2,3,1,0,2});

    type::vector<Index> indexToRemove{1,0};
    PointContainer.m_data.remove(indexToRemove);

    EXPECT_EQ(3,PointContainer.m_data.getValue().size());
    EXPECT_EQ(0,PointContainer.m_data.getValue()[0]);
    EXPECT_EQ(1,PointContainer.m_data.getValue()[1]);
    EXPECT_EQ(0,PointContainer.m_data.getValue()[2]);

}


TEST(TopologySubsetIndices_test, swapPoints)
{
    SimplePointTopology PointContainer(4);
    PointContainer.init();
    PointContainer.m_data.setValue({0,2,3,1,0,2});

    PointContainer.m_data.swap(0,5);

    EXPECT_EQ(6,PointContainer.m_data.getValue().size());
    EXPECT_EQ(2,PointContainer.m_data.getValue()[0]);
    EXPECT_EQ(2,PointContainer.m_data.getValue()[1]);
    EXPECT_EQ(3,PointContainer.m_data.getValue()[2]);
    EXPECT_EQ(1,PointContainer.m_data.getValue()[3]);
    EXPECT_EQ(0,PointContainer.m_data.getValue()[4]);
    EXPECT_EQ(0,PointContainer.m_data.getValue()[5]);
}

TEST(TopologySubsetIndices_test, renumber)
{
    SimplePointTopology PointContainer(4);
    PointContainer.init();
    PointContainer.m_data.setValue({0,2,3,1,0,2});

    PointContainer.m_data.renumber({5,2,3,4,1,0});

    EXPECT_EQ(6,PointContainer.m_data.getValue().size());
    EXPECT_EQ(2,PointContainer.m_data.getValue()[0]);
    EXPECT_EQ(3,PointContainer.m_data.getValue()[1]);
    EXPECT_EQ(1,PointContainer.m_data.getValue()[2]);
    EXPECT_EQ(0,PointContainer.m_data.getValue()[3]);
    EXPECT_EQ(2,PointContainer.m_data.getValue()[4]);
    EXPECT_EQ(0,PointContainer.m_data.getValue()[5]);

}


}
