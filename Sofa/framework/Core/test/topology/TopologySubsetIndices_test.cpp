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
    {
        m_points.reserve(size);
        for(unsigned i=0; i<size; ++i)
        {
            m_points.push_back(i);
        }
    }

    virtual const SeqEdges& getEdges() {}
    virtual const SeqTriangles& getTriangles() {}
    virtual const SeqQuads& getQuads() {}
    virtual const SeqTetrahedra& getTetrahedra() {}
    virtual const SeqHexahedra& getHexahedra() {}

    virtual sofa::geometry::ElementType getTopologyType() const
    {
        return sofa::geometry::ElementType::POINT;
    }


    virtual Size getNbPoints() const { return m_points.size(); }

    type::vector<sofa::Index> m_points;
};

TEST(TopologySubsetIndices_test, removePoints)
{
    const EdgeSetTopologyContainer::SPtr edgeContainer = sofa::core::objectmodel::New< EdgeSetTopologyContainer >();


    sofa::core::topology::BaseTopologyData<type::vector<Index>>::InitData initData ;
    TopologySubsetIndices data(initData);

    data.setValue({0,2,3,1,0,2});

    type::vector<Index> indexToRemove{1,0};
    data.remove(indexToRemove);

    EXPECT_EQ(0,data.getValue()[0]);
    EXPECT_EQ(1,data.getValue()[1]);
    EXPECT_EQ(0,data.getValue()[2]);

}

}
