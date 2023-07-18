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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <gtest/gtest.h>

namespace sofa
{

using core::topology::BaseMeshTopology;

template<class Container>
bool testInvalidContent(const Container& container)
{
    return std::all_of(container.begin(), container.end(), [](const auto id) { return id == sofa::InvalidID;});
}

TEST(BaseMeshTopology, invalidContainers)
{
    EXPECT_TRUE(testInvalidContent(BaseMeshTopology::InvalidEdgesInTriangles));
    EXPECT_TRUE(testInvalidContent(BaseMeshTopology::InvalidEdgesInQuad));
    EXPECT_TRUE(testInvalidContent(BaseMeshTopology::InvalidTrianglesInTetrahedron));
    EXPECT_TRUE(testInvalidContent(BaseMeshTopology::InvalidEdgesInTetrahedron));
    EXPECT_TRUE(testInvalidContent(BaseMeshTopology::InvalidQuadsInHexahedron));
    EXPECT_TRUE(testInvalidContent(BaseMeshTopology::InvalidEdgesInHexahedron));
}

}
